/*
* Copyright (c) Galib Arrieta (aka lumbermixalot@github, aka galibzon@github).
*
* SPDX-License-Identifier: Apache-2.0 OR MIT
*
*/

#include <AzCore/Name/NameDictionary.h>
#include <AzCore/Console/IConsole.h>

#include <Atom/RHI/DrawPacketBuilder.h>
#include <Atom/RHI.Reflect/InputStreamLayoutBuilder.h>

#include <Atom/RPI.Public/Image/AttachmentImagePool.h>
#include <Atom/RPI.Public/RenderPipeline.h>

#include <Atom/RPI.Public/RPIUtils.h>
#include <Atom/RPI.Reflect/Asset/AssetUtils.h> // FIXME: Try removing

#include <Atom/RPI.Public/Pass/PassFilter.h>
#include <Atom/RPI.Public/Pass/RasterPass.h>
#include <Atom/RPI.Public/Shader/Shader.h>
#include <Atom/RPI.Public/Scene.h>
#include <Atom/RPI.Public/View.h>
#include <Atom/RPI.Public/ViewportContext.h>
#include <Atom/RPI.Public/Image/ImageSystemInterface.h>

#include <Renderer/Passes/CloudscapeComputePass.h>
#include <Renderer/Passes/CloudscapeRasterPass.h>
// #include <Renderer/Passes/DepthBufferCopyPass.h>
#include "CloudscapeFeatureProcessor.h"

namespace VolumetricClouds
{
    AZ_CVAR(bool, r_cloudsEnabled, true, nullptr, AZ::ConsoleFunctorFlags::Null, "Render the volumetric cloudscape (does not alter scene settings).");
    void CloudscapeFeatureProcessor::Reflect(AZ::ReflectContext* context)
    {
        if (auto* serializeContext = azrtti_cast<AZ::SerializeContext*>(context))
        {
            serializeContext
                ->Class<CloudscapeFeatureProcessor, AZ::RPI::FeatureProcessor>()
                ->Version(1);
        }
    }

    /////////////////////////////////////////////////////////////////////////////
    //! AZ::RPI::FeatureProcessor overrides START ...
    void CloudscapeFeatureProcessor::Activate()
    {
        ActivateInternal();
    }

    void CloudscapeFeatureProcessor::RemovePasses()
    {
        // The render pipeline may already have detached its passes during shutdown.
        // Only queue passes that still belong to a parent, and handle partial setup.
        if (m_cloudscapeComputePass && m_cloudscapeComputePass->GetParent())
        {
            m_cloudscapeComputePass->QueueForRemoval();
        }
        if (m_cloudscapeReprojectionPass && m_cloudscapeReprojectionPass->GetParent())
        {
            m_cloudscapeReprojectionPass->QueueForRemoval();
        }
        if (m_cloudscapeRenderPass && m_cloudscapeRenderPass->GetParent())
        {
            m_cloudscapeRenderPass->QueueForRemoval();
        }
        m_cloudscapeComputePass = nullptr;
        m_cloudscapeReprojectionPass = nullptr;
        m_cloudscapeRenderPass = nullptr;

        m_renderPipeline = nullptr;
    }

    void CloudscapeFeatureProcessor::OnRenderPipelineRemoved(AZ::RPI::RenderPipeline* pipeline)
    {
        if (pipeline == m_renderPipeline)
        {
            RemovePasses();
        }
    }

    void CloudscapeFeatureProcessor::Deactivate()
    {
        DisableSceneNotification();
        RemovePasses();
        m_cloudOutput0.reset();
        m_cloudOutput1.reset();
        m_shaderConstantData = {};
        m_hasShaderConstantData = false;
        m_viewportSize = { 0,0 };
    }

    void CloudscapeFeatureProcessor::Simulate(const SimulatePacket&)
    {
        if (m_cloudscapeComputePass && m_cloudscapeReprojectionPass && m_cloudscapeRenderPass
            && m_cloudscapeReprojectionPass->GetShaderResourceGroup())
        {
            const bool ready = r_cloudsEnabled && m_hasShaderConstantData
                && m_shaderConstantData.m_lowFrequencyNoiseTexture
                && m_shaderConstantData.m_highFrequencyNoiseTexture && m_shaderConstantData.m_weatherMap;
            m_cloudscapeComputePass->SetEnabled(ready);
            m_cloudscapeReprojectionPass->SetEnabled(ready);
            m_cloudscapeRenderPass->SetEnabled(ready);
            if (!ready)
            {
                m_frameCounter = 0;
                return;
            }
            m_cloudscapeComputePass->UpdateFrameCounter(m_frameCounter);

            const auto& passSrg = m_cloudscapeReprojectionPass->GetShaderResourceGroup();
            const uint32_t pixelIndex4x4 = m_frameCounter % 16;
            passSrg->SetConstant(m_pixelIndex4x4Index, pixelIndex4x4);
            if (auto view = m_renderPipeline->GetDefaultView())
            {
                const auto camera = view->GetViewToWorldMatrix();
                const bool cameraCut = camera.GetTranslation().GetDistanceSq(m_previousCamera.GetTranslation()) > 100.0f
                    || camera.GetBasisY().Dot(m_previousCamera.GetBasisY()) < 0.95f;
                const uint32_t historyValid = m_frameCounter > 0 && !m_resetHistory && !cameraCut;
                passSrg->SetConstant(m_historyValidIndex, historyValid);
                m_previousCamera = camera;
                m_resetHistory = false;
            }

            m_cloudscapeRenderPass->UpdateFrameCounter(m_frameCounter);
            
            m_frameCounter++;
        }
    }

    void CloudscapeFeatureProcessor::AddRenderPasses([[maybe_unused]] AZ::RPI::RenderPipeline* renderPipeline)
    {
        // Only attach to the main viewport pipeline. Preview/compute pipelines do
        // not have the required depth/motion-vector/transparent passes.
        if (m_renderPipeline || !m_cloudOutput0 || !m_cloudOutput1)
        {
            return;
        }
        for (const char* anchor : { "DepthPrePass", "MotionVectorPass", "TransparentPass" })
        {
            if (!AZ::RPI::PassSystemInterface::Get()->FindFirstPass(
                AZ::RPI::PassFilter::CreateWithPassName(AZ::Name(anchor), renderPipeline)))
            {
                return;
            }
        }
        m_renderPipeline = renderPipeline;
        m_frameCounter = 0;
        // Get the pass requests to create passes from the asset
        AddPassRequestToRenderPipeline(renderPipeline, "Passes/CloudscapeComputePassRequest.azasset", "DepthPrePass", false /*before*/);
        // Hold a reference to the compute pass
        {
            const auto passName = AZ::Name("CloudscapeComputePass");
            AZ::RPI::PassFilter passFilter = AZ::RPI::PassFilter::CreateWithPassName(passName, renderPipeline);
            AZ::RPI::Pass* existingPass = AZ::RPI::PassSystemInterface::Get()->FindFirstPass(passFilter);
            m_cloudscapeComputePass = azrtti_cast<CloudscapeComputePass*>(existingPass);
            if (!m_cloudscapeComputePass)
            {
                AZ_Error(LogName, false, "%s Failed to find as RenderPass: %s", __FUNCTION__, passName.GetCStr());
                RemovePasses();
                return;
            }

            if (m_hasShaderConstantData)
            {
                m_cloudscapeComputePass->UpdateShaderConstantData(m_shaderConstantData);
            }
        }

        AddPassRequestToRenderPipeline(renderPipeline, "Passes/CloudscapeReprojectionComputePassRequest.azasset", "MotionVectorPass", false /*before*/);
        // Hold a reference to the compute pass
        {
            const auto passName = AZ::Name("CloudscapeReprojectionComputePass");
            AZ::RPI::PassFilter passFilter = AZ::RPI::PassFilter::CreateWithPassName(passName, renderPipeline);
            AZ::RPI::Pass* existingPass = AZ::RPI::PassSystemInterface::Get()->FindFirstPass(passFilter);
            m_cloudscapeReprojectionPass = azrtti_cast<AZ::RPI::ComputePass*>(existingPass);
            if (!m_cloudscapeReprojectionPass)
            {
                AZ_Error(LogName, false, "%s Failed to find as RenderPass: %s", __FUNCTION__, passName.GetCStr());
                RemovePasses();
                return;
            }
            m_cloudscapeReprojectionPass->SetTargetThreadCounts(m_viewportSize.m_width, m_viewportSize.m_height, 1);
        }


        AddPassRequestToRenderPipeline(renderPipeline, "Passes/CloudscapeRasterPassRequest.azasset", "TransparentPass", true /*before*/);
        // Hold a reference to the render pass
        {
            const auto passName = AZ::Name("CloudscapeRasterPass");
            AZ::RPI::PassFilter passFilter = AZ::RPI::PassFilter::CreateWithPassName(passName, renderPipeline);
            AZ::RPI::Pass* existingPass = AZ::RPI::PassSystemInterface::Get()->FindFirstPass(passFilter);
            m_cloudscapeRenderPass = azrtti_cast<CloudscapeRasterPass*>(existingPass);
            if (!m_cloudscapeRenderPass)
            {
                AZ_Error(LogName, false, "%s Failed to find as RenderPass: %s", __FUNCTION__, passName.GetCStr());
                RemovePasses();
                return;
            }
        }

    }

    //! AZ::RPI::FeatureProcessor overrides END ...
    /////////////////////////////////////////////////////////////////////////////


    /////////////////////////////////////////////////////////////////////
    //! Functions called by CloudscapeComponentController START
    void CloudscapeFeatureProcessor::UpdateShaderConstantData(const CloudscapeShaderConstantData& shaderData)
    {
        m_resetHistory = m_resetHistory || !m_hasShaderConstantData || m_shaderConstantData != shaderData
            || m_shaderConstantData.m_lowFrequencyNoiseTexture != shaderData.m_lowFrequencyNoiseTexture
            || m_shaderConstantData.m_highFrequencyNoiseTexture != shaderData.m_highFrequencyNoiseTexture
            || m_shaderConstantData.m_weatherMap != shaderData.m_weatherMap;
        m_shaderConstantData = shaderData;
        m_hasShaderConstantData = true;
        if (m_cloudscapeComputePass)
        {
            m_cloudscapeComputePass->UpdateShaderConstantData(shaderData);
        }
    }

    //! Functions called by CloudscapeComponentController END
    /////////////////////////////////////////////////////////////////////


    void CloudscapeFeatureProcessor::ActivateInternal()
    {
        auto viewportContextInterface = AZ::Interface<AZ::RPI::ViewportContextRequestsInterface>::Get();
        if (!viewportContextInterface)
        {
            return;
        }
        auto viewportContext = viewportContextInterface->GetViewportContextByScene(GetParentScene());
        if (!viewportContext)
        {
            AZ_Warning(LogName, false, "No viewport is available for the cloud scene.");
            return;
        }
        m_viewportSize = viewportContext->GetViewportSize();
        m_viewportSize.m_width = AZStd::max(1u, m_viewportSize.m_width);
        m_viewportSize.m_height = AZStd::max(1u, m_viewportSize.m_height);

        m_cloudOutput0 = CreateCloudscapeOutputAttachment(AZ::Name("CloudscapeOutput0"), m_viewportSize);
        AZ_Assert(!!m_cloudOutput0, "Failed to create CloudscapeOutput0");
        m_cloudOutput1 = CreateCloudscapeOutputAttachment(AZ::Name("CloudscapeOutput1"), m_viewportSize);
        AZ_Assert(!!m_cloudOutput1, "Failed to create CloudscapeOutput1");

        DisableSceneNotification();
        EnableSceneNotification();
    }


    AZ::Data::Instance<AZ::RPI::AttachmentImage> CloudscapeFeatureProcessor::CreateCloudscapeOutputAttachment(const AZ::Name& attachmentName
        , const AzFramework::WindowSize attachmentSize) const
    {
        AZ::RHI::ImageDescriptor imageDesc = AZ::RHI::ImageDescriptor::Create2D(
            // Radiance and temporal history must retain values above 1 until the scene display transform.
            AZ::RHI::ImageBindFlags::ShaderReadWrite, attachmentSize.m_width, attachmentSize.m_height, AZ::RHI::Format::R16G16B16A16_FLOAT);
        AZ::RHI::ClearValue clearValue = AZ::RHI::ClearValue::CreateVector4Float(0, 0, 0, 0);
        AZ::Data::Instance<AZ::RPI::AttachmentImagePool> pool = AZ::RPI::ImageSystemInterface::Get()->GetSystemAttachmentPool();
        return AZ::RPI::AttachmentImage::Create(*pool.get(), imageDesc, attachmentName, &clearValue, nullptr);
    }

} // namespace VolumetricClouds
