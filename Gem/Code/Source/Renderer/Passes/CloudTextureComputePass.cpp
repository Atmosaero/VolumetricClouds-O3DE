/*
* Copyright (c) Galib Arrieta (aka lumbermixalot@github, aka galibzon@github).
*
* SPDX-License-Identifier: Apache-2.0 OR MIT
*
*/

#include <Atom/RHI/FrameGraphAttachmentInterface.h>
#include <Atom/RHI/FrameGraphBuilder.h>
#include <Atom/RHI/CommandList.h>

#include <Atom/RPI.Public/Pass/PassUtils.h>
#include <Atom/RPI.Public/RenderPipeline.h>
#include <Atom/RPI.Public/RPIUtils.h>
#include <Atom/RPI.Public/Scene.h>
#include <Atom/RPI.Public/View.h>
#include <Atom/RPI.Reflect/Shader/ShaderAsset.h>

#include "CloudTextureComputePass.h"


namespace VolumetricClouds
{
    AZ::RPI::Ptr<CloudTextureComputePass> CloudTextureComputePass::Create(const AZ::RPI::PassDescriptor& descriptor)
    {
        AZ::RPI::Ptr<CloudTextureComputePass> pass = aznew CloudTextureComputePass(descriptor);
        return pass;
    }

    CloudTextureComputePass::CloudTextureComputePass(const AZ::RPI::PassDescriptor& descriptor)
        : AZ::RPI::ComputePass(descriptor)
    {
    }

    uint16_t CloudTextureComputePass::CalculateMipCount(uint32_t pixelSize)
    {
        return static_cast<uint16_t>(log2(pixelSize) - log2(MIN_PIXEL_SIZE)) + 1;
    }

    void CloudTextureComputePass::BuildInternal()
    {
        if (!m_texture3DAttachment)
        {
            // This is OK, because BuildInternal is always called when the pipeline is created,
            // and at this time there's no data given to this pass.
            // Later when ScenePtr->AddRenderPipeline is called, this function is called again,
            // but at this time m_texture3DAttachment was already setup and the attachment binding are properly
            // initialized.
            return;
        }

        const uint32_t mipSize = m_computeData.m_pixelSize >> m_mipLevel;
        auto output = FindAttachmentBinding(AZ::Name("OutputMip0"));
        AZ_Assert(output, "Missing noise output slot");
        output->m_shaderInputName = AZ::Name("m_outputMip");
        output->m_unifiedScopeDesc.SetAsImage(AZ::RHI::ImageViewDescriptor::Create3D(
            m_texture3DAttachment->GetDescriptor().m_format, m_mipLevel, m_mipLevel, 0, static_cast<uint16_t>(mipSize - 1)));
        AttachImageToSlot(AZ::Name("OutputMip0"), m_texture3DAttachment);

        if (m_mipLevel > 0)
        {
            // Only filtering passes have a source. Mip 0 and unused passes must
            // not declare an unbound required input in pass validation.
            AZ::RPI::PassSlot inputSlot;
            inputSlot.m_name = AZ::Name("InputMip");
            inputSlot.m_slotType = AZ::RPI::PassSlotType::Input;
            inputSlot.m_scopeAttachmentUsage = AZ::RHI::ScopeAttachmentUsage::Shader;
            inputSlot.m_shaderInputName = AZ::Name("m_sourceMip");
            AddAttachmentBinding(AZ::RPI::PassAttachmentBinding(inputSlot));
            auto input = FindAttachmentBinding(AZ::Name("InputMip"));
            AZ_Assert(input, "Missing noise input slot");
            input->m_shaderInputName = AZ::Name("m_sourceMip");
            input->m_unifiedScopeDesc.SetAsImage(AZ::RHI::ImageViewDescriptor::Create3D(
                m_texture3DAttachment->GetDescriptor().m_format, m_mipLevel - 1, m_mipLevel - 1, 0, static_cast<uint16_t>(mipSize * 2 - 1)));
            AttachImageToSlot(AZ::Name("InputMip"), m_texture3DAttachment);
        }
        SetTargetThreadCounts(mipSize, mipSize, mipSize);
    }

    void CloudTextureComputePass::FrameBeginInternal(FramePrepareParams params)
    {
        AZ::RPI::RenderPass::FrameBeginInternal(params);
    }

    void CloudTextureComputePass::FrameEndInternal()
    {
        if (!m_texture3DAttachment)
        {
            return;
        }

        m_isFinished = true;

        SetEnabled(false);
    }

    void CloudTextureComputePass::SetupFrameGraphDependencies(AZ::RHI::FrameGraphInterface frameGraph)
    {
        if (!m_texture3DAttachment)
        {
            AZ_Error(LogName, false, "Where is the texture3DAttachment?");
            return;
        }

        // Bread crumbs:
        // Typically when defining Image Attachments in *.pass json files, along with "Connections"
        // AND if the pass owns the persistent attachment the Import and Use are done automatically
        // by the base Pass class SetupFrameGraphDependencies.
        AZ::RHI::FrameGraphAttachmentInterface attachmentDatabase = frameGraph.GetAttachmentDatabase();
        if (!attachmentDatabase.IsAttachmentValid(m_texture3DAttachment->GetAttachmentId()))
        {
            attachmentDatabase.ImportImage(m_texture3DAttachment->GetAttachmentId(), m_texture3DAttachment->GetRHIImage());
        }

        // REMARK:
        // Commented this block because it is redundant because AZ::RPI::ComputePass::SetupFrameGraphDependencies
        // already does the same.
        // 
        // AZ::RHI::ImageScopeAttachmentDescriptor desc;
        // desc.m_imageViewDescriptor = request.m_cloudTextureAttachment->GetImageView()->GetDescriptor();
        // desc.m_loadStoreAction.m_loadAction = AZ::RHI::AttachmentLoadAction::Load;
        // desc.m_attachmentId = request.m_cloudTextureAttachment->GetAttachmentId();
        // // From the point of view of this compute pass we only need Write access, but this attachment
        // // will be used later by CloudTexturesFeatureProcessor as a Read texture.
        // frameGraph.UseShaderAttachment(desc, AZ::RHI::ScopeAttachmentAccess::ReadWrite);

        AZ::RPI::ComputePass::SetupFrameGraphDependencies(frameGraph);
    }

    void CloudTextureComputePass::CompileResources(const AZ::RHI::FrameGraphCompileContext& context)
    {
        if (m_texture3DAttachment)
        {
            const auto& computeData = m_computeData;

            m_shaderResourceGroup->SetConstant(m_frequencyIndex, computeData.m_frequency);

            m_shaderResourceGroup->SetConstant(m_perlinOctavesIndex,   computeData.m_perlinOctaves);
            m_shaderResourceGroup->SetConstant(m_perlinGainIndex,      computeData.m_perlinGain);
            m_shaderResourceGroup->SetConstant(m_perlinAmplitudeIndex, computeData.m_perlinAmplitude);

            m_shaderResourceGroup->SetConstant(m_worleyOctavesIndex,   computeData.m_worleyOctaves);
            m_shaderResourceGroup->SetConstant(m_worleyGainIndex,      computeData.m_worleyGain);
            m_shaderResourceGroup->SetConstant(m_worleyAmplitudeIndex, computeData.m_worleyAmplitude);

            m_shaderResourceGroup->SetConstant(m_pixelSizeIndex, computeData.m_pixelSize);
            m_shaderResourceGroup->SetConstant(m_mipLevelIndex, uint32_t(m_mipLevel));

        }
        AZ::RPI::ComputePass::CompileResources(context);
    }

    bool CloudTextureComputePass::IsEnabled() const
    {
        if (!AZ::RPI::Pass::IsEnabled())
        {
            return false;
        }

        return !m_isFinished && m_texture3DAttachment;
    }

    bool CloudTextureComputePass::SetRenderData(AZ::Data::Instance<AZ::RPI::AttachmentImage> texture3DAttachment,
        CloudTextureComputeData computeData, uint16_t mipLevel)
    {
        if (m_isFinished)
        {
            AZ_Error(LogName, false, "This function can not be called after the pass is finished!");
            return false;
        }

        const auto pixelSize = computeData.m_pixelSize;
        if (!texture3DAttachment || pixelSize < MIN_PIXEL_SIZE || pixelSize > MAX_PIXEL_SIZE || (pixelSize & (pixelSize - 1)) != 0)
        {
            AZ_Error(LogName, false, "This pass can not generate noise textures smaller than %u or larger than %u pixels. Got %u pixels.\n",
                MIN_PIXEL_SIZE, MAX_PIXEL_SIZE, pixelSize);
            return false;
        }

        // Make sure the Texture3D was memory allocated properly (with expected MipMap count, etc).
        const auto expectedMipsCount = CalculateMipCount(pixelSize);
        const auto &imageDesc = texture3DAttachment->GetDescriptor();
        if (imageDesc.m_mipLevels != expectedMipsCount || mipLevel >= expectedMipsCount)
        {
            AZ_Error(LogName, false, "For pixel size %u, Expected Mips count %u in attachment image don't match. Got %u\n",
                pixelSize, expectedMipsCount, imageDesc.m_mipLevels);
            return false;
        }

        m_texture3DAttachment = texture3DAttachment;
        m_computeData = computeData;
        m_mipLevel = mipLevel;

        SetEnabled(true);
        return true;
    }

} // namespace VolumetricClouds
