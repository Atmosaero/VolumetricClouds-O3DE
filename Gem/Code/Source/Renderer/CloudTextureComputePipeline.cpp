/*
* Copyright (c) Galib Arrieta (aka lumbermixalot@github, aka galibzon@github).
*
* SPDX-License-Identifier: Apache-2.0 OR MIT
*
*/

#include <Atom/RPI.Public/Pass/Pass.h>
#include <Atom/RPI.Public/Pass/PassSystemInterface.h>
#include <Atom/RPI.Public/Pass/PassFilter.h>
#include <Atom/RPI.Public/RenderPipeline.h>
#include <Atom/RPI.Public/View.h>
#include <Atom/RPI.Reflect/Asset/AssetUtils.h>
#include <Atom/RPI.Reflect/System/AnyAsset.h>

#include <Renderer/Passes/CloudTextureComputePass.h>
#include "CloudTextureComputePipeline.h"

namespace VolumetricClouds
{
    CloudTextureComputePipeline::RenderTaskId CloudTextureComputePipeline::m_renderTaskCounter = 0;

    CloudTextureComputePipeline::RenderTaskId CloudTextureComputePipeline::StartTextureCompute(AZ::RPI::Scene* scene, AZ::Data::Instance<AZ::RPI::AttachmentImage> texture3DAttachment,
        const CloudTextureComputeData& computeData, CloudTextureRenderCallback callback, bool withAttachmentReadback)
    {
        m_scene = scene;
        AZ_Assert(m_isRendering == false, "CloudTextureComputePipeline::StartRender called while a noise Texture render was already in progress");
        if (m_isRendering)
        {
            return 0;
        }
    
        m_callback = AZStd::move(callback);
        if (!m_scene || !texture3DAttachment)
        {
            return 0;
        }

        AZ::Data::Asset<AZ::RPI::AnyAsset> pipelineAsset = AZ::RPI::AssetUtils::LoadAssetByProductPath<AZ::RPI::AnyAsset>(PipelineDescriptorAssetPath, AZ::RPI::AssetUtils::TraceLevel::Error);
        if (!pipelineAsset.IsReady())
        {
            AZ_Assert(false, "Failed to load pipeline asset at %s", PipelineDescriptorAssetPath);
            AZ_Error(LogName, false, "Failed to load pipeline asset at %s", PipelineDescriptorAssetPath);
            return 0;
        }

        const AZ::RPI::RenderPipelineDescriptor* tmpRenderPipelineDescriptor = AZ::RPI::GetDataFromAnyAsset<AZ::RPI::RenderPipelineDescriptor>(pipelineAsset);
        AZ_Assert(!!tmpRenderPipelineDescriptor, "Couldn't read asset %s as RenderPipelineDescriptor", PipelineDescriptorAssetPath);
        if (!tmpRenderPipelineDescriptor)
        {
            return 0;
        }
        AZ::RPI::RenderPipelineDescriptor renderPipelineDescriptor = *tmpRenderPipelineDescriptor;
        // Define a unique name for the pipeline within the scene.
        m_renderTaskCounter++;
        m_renderTaskId = m_renderTaskCounter;
        renderPipelineDescriptor.m_name = AZStd::string::format("CloudTexturePipeline_%u", m_renderTaskId);

        AZ::RPI::RenderPipelinePtr renderPipeline = AZ::RPI::RenderPipeline::CreateRenderPipeline(renderPipelineDescriptor);
        if (!renderPipeline)
        {
            return 0;
        }
        m_renderPipelineId = renderPipeline->GetId();

        // Separate scopes are required: a dispatch-wide barrier cannot synchronize
        // mip generation across thread groups. The last pass owns completion/readback.
        const uint16_t mipCount = CloudTextureComputePass::CalculateMipCount(computeData.m_pixelSize);
        for (uint16_t mip = 0; mip < mipCount; ++mip)
        {
            const AZ::Name passName(mip == 0 ? AZStd::string("CloudTextureComputePass") : AZStd::string::format("CloudTextureMip%u", mip));
            const auto filter = AZ::RPI::PassFilter::CreateWithPassName(passName, renderPipeline.get());
            auto pass = azrtti_cast<CloudTextureComputePass*>(AZ::RPI::PassSystemInterface::Get()->FindFirstPass(filter));
            if (!pass || !pass->SetRenderData(texture3DAttachment, computeData, mip))
            {
                AZ_Error(LogName, false, "Failed to initialize noise mip pass %s", passName.GetCStr());
                return 0;
            }
            m_textureComputePass = pass;
        }

        // Add the pipeline to the scene
        m_scene->AddRenderPipeline(renderPipeline);
        m_isRendering = true;

        if (withAttachmentReadback)
        {
            // Setup the attachment readback if the user needs to read the Texture3D from GPU to CPU memory.
            SetupAttachmentReadback(computeData.m_pixelSize);
        }

        return m_renderTaskId;
    }
    
    void CloudTextureComputePipeline::CheckAndRemovePipeline()
    {
        if (m_isRendering && m_textureComputePass && m_textureComputePass->IsFinished())
        {
            if (m_readbackState && !m_readbackState->m_complete.load(AZStd::memory_order_acquire))
            {
                return;
            }

            auto callback = AZStd::move(m_callback);
            if (callback)
            {
                const AZStd::vector<CloudTextureSubresourceReadback> empty;
                callback(m_renderTaskId, m_readbackState ? m_readbackState->m_data : empty);
            }

            m_isRendering = false;
    
            // remove the cubemap pipeline
            // Note: this must not be called in the scope of a feature processor Simulate or Render to avoid a race condition with other feature processors
            m_scene->RemoveRenderPipeline(m_renderPipelineId);
            m_attachmentsReadback.reset();
            m_readbackState.reset();
            m_textureComputePass = nullptr;
        }
    }

    void CloudTextureComputePipeline::Cancel()
    {
        m_callback = {};
        if (m_isRendering && m_scene)
        {
            m_scene->RemoveRenderPipeline(m_renderPipelineId);
        }
        m_isRendering = false;
        m_textureComputePass = nullptr;
        m_attachmentsReadback.reset();
        m_readbackState.reset();
        m_scene = nullptr;
    }

    void CloudTextureComputePipeline::SetupAttachmentReadback(uint32_t pixelSize)
    {
        AZStd::fixed_string<128> scope_name = AZStd::fixed_string<128>::format("Texture3DCapture_%u", m_renderTaskId);
        m_attachmentsReadback = AZStd::make_shared<AZ::RPI::AttachmentReadback>(AZ::RHI::ScopeId{ scope_name });
        m_readbackState = AZStd::make_shared<ReadbackState>();
        m_attachmentsReadback->SetCallback([state = m_readbackState](const AZ::RPI::AttachmentReadback::ReadbackResult& result)
        {
            for (const auto& mip : result.m_mipDataBuffers)
            {
                state->m_data.push_back({mip.m_mipBuffer, mip.m_mipInfo.m_slice, mip.m_mipInfo.m_size});
            }
            state->m_complete.store(true, AZStd::memory_order_release);
        });
        m_attachmentsReadback->SetUserIdentifier(m_renderTaskId);

        AZ::Name slotName("OutputMip0");

        const auto mipsCount = CloudTextureComputePass::CalculateMipCount(pixelSize);
        const uint16_t mipSliceMax = mipsCount - 1;
        AZ::RHI::ImageSubresourceRange mipsRange(0 /*mipSliceMin*/, mipSliceMax, 0, 0);
        const bool result = m_textureComputePass->ReadbackAttachment(m_attachmentsReadback, m_renderTaskId,
            slotName, AZ::RPI::PassAttachmentReadbackOption::Output, &mipsRange);
        AZ_Error(LogName, result, "%s Failed to initialize ReadbackAttachment\n", __FUNCTION__);
        if (!result)
        {
            m_attachmentsReadback.reset();
            m_readbackState.reset();
        }
    }
    
} // namespace VolumetricClouds
