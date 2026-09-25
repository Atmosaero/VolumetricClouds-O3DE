/*
* Copyright (c) Galib Arrieta (aka lumbermixalot@github, aka galibzon@github).
*
* SPDX-License-Identifier: Apache-2.0 OR MIT
*
*/

#include <Atom/RPI.Public/Image/AttachmentImagePool.h>
#include <Atom/RPI.Public/RenderPipeline.h>
#include <Atom/RPI.Public/Scene.h>
#include <Atom/RPI.Public/Image/ImageSystemInterface.h>
#include <AzCore/Console/IConsole.h>

#include <Renderer/Passes/CloudTextureComputePass.h>
#include "CloudTexturesComputeFeatureProcessor.h"

namespace VolumetricClouds
{
    AZ_CVAR(bool, r_cloudsValidateNoiseMips, false, nullptr, AZ::ConsoleFunctorFlags::Null,
        "Read generated noise back from the GPU and validate the filtered mip chain (diagnostic only).");

    namespace
    {
        bool ValidateNoiseMips(const AZStd::vector<CloudTextureComputePipeline::CloudTextureSubresourceReadback>& results,
            uint32_t pixelSize)
        {
            const auto mipCount = CloudTextureComputePass::CalculateMipCount(pixelSize);
            if (results.size() != mipCount)
            {
                return false;
            }
            const AZStd::vector<uint8_t>* previous = nullptr;
            for (uint16_t mip = 0; mip < mipCount; ++mip)
            {
                const uint32_t size = pixelSize >> mip;
                const CloudTextureComputePipeline::CloudTextureSubresourceReadback* level = nullptr;
                for (const auto& result : results)
                {
                    if (result.m_mipSlice == mip)
                    {
                        level = &result;
                        break;
                    }
                }
                if (!level || !level->m_dataBuffer || level->m_mipSize != AZ::RHI::Size(size, size, size) ||
                    level->m_dataBuffer->size() != size_t(size) * size * size * 4)
                {
                    return false;
                }
                if (previous)
                {
                    const uint32_t sourceSize = size * 2;
                    for (uint32_t z = 0; z < size; ++z)
                    {
                        for (uint32_t y = 0; y < size; ++y)
                        {
                            for (uint32_t x = 0; x < size; ++x)
                            {
                                for (uint32_t channel = 0; channel < 4; ++channel)
                                {
                                    int sum = 0;
                                    for (uint32_t dz = 0; dz < 2; ++dz)
                                    {
                                        for (uint32_t dy = 0; dy < 2; ++dy)
                                        {
                                            for (uint32_t dx = 0; dx < 2; ++dx)
                                            {
                                                const size_t index = ((size_t(z * 2 + dz) * sourceSize + y * 2 + dy) *
                                                    sourceSize + x * 2 + dx) * 4 + channel;
                                                sum += (*previous)[index];
                                            }
                                        }
                                    }
                                    const int actual = (*level->m_dataBuffer)[((size_t(z) * size + y) * size + x) * 4 + channel];
                                    // Allow one UNORM LSB, including implementation-dependent tie rounding.
                                    if (AZStd::abs(actual * 8 - sum) > 8)
                                    {
                                        return false;
                                    }
                                }
                            }
                        }
                    }
                }
                previous = level->m_dataBuffer.get();
            }
            return true;
        }
    }

    void CloudTexturesComputeFeatureProcessor::Reflect(AZ::ReflectContext* context)
    {
        if (auto* serializeContext = azrtti_cast<AZ::SerializeContext*>(context))
        {
            serializeContext
                ->Class<CloudTexturesComputeFeatureProcessor, AZ::RPI::FeatureProcessor>()
                ->Version(1);
        }
    }

    /////////////////////////////////////////////////////////////////////////////
    //! AZ::RPI::FeatureProcessor overrides START ...
    void CloudTexturesComputeFeatureProcessor::Activate()
    {
        ActivateComputeScene();
    }

    void CloudTexturesComputeFeatureProcessor::ActivateComputeScene()
    {
        AZ_Assert(!m_computeScene, "Compute Scene was already activated!");

        AZ::RPI::SceneDescriptor sceneDesc;
        m_computeScene = AZ::RPI::Scene::CreateScene(sceneDesc);

        // Currently the scene has to be activated after render pipeline was added so some feature processors (i.e. imgui) can be initialized properly 
        // with pipeline's pass information. 
        m_computeScene->Activate();

        AZ::RPI::RPISystemInterface::Get()->RegisterScene(m_computeScene);
    }

    void CloudTexturesComputeFeatureProcessor::Deactivate()
    {
        if (m_currentCloudTextureComputeTask)
        {
            m_currentCloudTextureComputeTask->Cancel();
            m_currentCloudTextureComputeTask.reset();
        }
        m_computeRequests.clear();
        m_textureCache.clear();
        m_cachedBytes = 0;

        DeactivateComputeScene();
    }

    void CloudTexturesComputeFeatureProcessor::DeactivateComputeScene()
    {
        decltype(m_cloudTextureComputeTasks) tmpQueue;
        AZStd::swap(tmpQueue, m_cloudTextureComputeTasks);

        if (m_computeScene)
        {
            AZ::RPI::RPISystemInterface::Get()->UnregisterScene(m_computeScene);
            m_computeScene->Deactivate();
            m_computeScene = nullptr;
        }

    }


    void CloudTexturesComputeFeatureProcessor::OnRenderEnd()
    {
        // // The C++ 20 new way to do remove erase if.
        // AZStd::erase_if(m_cloudTextureComputeTasks, [](AZStd::shared_ptr<CloudTextureComputePipeline>& renderer) {
        //     renderer->CheckAndRemovePipeline();
        //     return !renderer->IsRenderingNoiseTexture();
        // });
        if (m_currentCloudTextureComputeTask)
        {
            m_currentCloudTextureComputeTask->CheckAndRemovePipeline();
            if (m_currentCloudTextureComputeTask->IsRenderingNoiseTexture())
            {
                return;
            }
            m_currentCloudTextureComputeTask.reset();
        }

        while (!m_cloudTextureComputeTasks.empty())
        {
            AZ::EntityId entityId = m_cloudTextureComputeTasks.front();
            m_cloudTextureComputeTasks.pop_front();
            if (!m_computeRequests.contains(entityId))
            {
                AZ_Info(LogName, "CloudTextureComputeRequest with entityId=%s already gone.\n", entityId.ToString().c_str());
                continue;
            }
            auto CloudTextureComputeRequest = m_computeRequests.at(entityId);
            m_currentCloudTextureComputeTask = CreateTextureComputeTask(CloudTextureComputeRequest);
            AZ_Info(LogName, "Created new compute task id=%u for entityId=%s.\n",
                CloudTextureComputeRequest->m_textureComputeTaskId, entityId.ToString().c_str());
            break;
        }

    }

    //! AZ::RPI::FeatureProcessor overrides END ...
    /////////////////////////////////////////////////////////////////////////////


    /////////////////////////////////////////////////////////////////////
    //! Functions called by CloudTextureComputeComponentController START
    bool CloudTexturesComputeFeatureProcessor::EnqueueComputeRequest(const AZ::EntityId& entityId
        , const CloudTextureComputeData& computeData
        , CloudTexturesComputeFeatureProcessor::TextureReadyEvent::Handler& readyHandler
        , CloudTexturesComputeFeatureProcessor::ReadbackEvent::Handler* readbackHandler)
    {
        const auto size = computeData.m_pixelSize;
        if (size < 16 || size > 256 || (size & (size - 1)) != 0)
        {
            AZ_Error(LogName, false, "Noise texture size must be a power of two in [16, 256], got %u.", size);
            return false;
        }
        if (!readbackHandler)
        {
            for (const auto& cached : m_textureCache)
            {
                if (cached.m_data == computeData)
                {
                    CancelComputeRequest(entityId);
                    TextureReadyEvent ready;
                    readyHandler.Disconnect();
                    readyHandler.Connect(ready);
                    ready.Signal(cached.m_image);
                    return true;
                }
            }
        }
        // One pending request per entity: edits replace stale work. Allocation is
        // delayed until dispatch, so sliders do not allocate a 3D image per event.
        CancelComputeRequest(entityId);
        auto request = AZStd::make_shared<CloudTextureComputeRequest>();
        request->m_computeData = computeData;
        request->m_withAttachmentReadback = readbackHandler != nullptr;
        readyHandler.Disconnect();
        readyHandler.Connect(request->m_readyEvent);
        if (readbackHandler)
        {
            readbackHandler->Disconnect();
            readbackHandler->Connect(request->m_readbackEvent);
        }
        m_computeRequests.emplace(entityId, AZStd::move(request));
        m_cloudTextureComputeTasks.push_back(entityId);
        return true;
    }

    void CloudTexturesComputeFeatureProcessor::CancelComputeRequest(const AZ::EntityId& entityId)
    {
        // In-flight GPU work finishes before its resources are released, but no
        // obsolete result is delivered to a deactivated/reconfigured component.
        auto found = m_computeRequests.find(entityId);
        if (found != m_computeRequests.end())
        {
            found->second->m_readyEvent.DisconnectAllHandlers();
            found->second->m_readbackEvent.DisconnectAllHandlers();
            m_computeRequests.erase(found);
        }
        AZStd::erase(m_cloudTextureComputeTasks, entityId);
    }
    //! Functions called by CloudscapeComponentController END
    /////////////////////////////////////////////////////////////////////

    AZ::Data::Instance<AZ::RPI::AttachmentImage> CloudTexturesComputeFeatureProcessor::CreateTexture3DAttachmentImage(uint32_t pixelSize)
    {
        AZ::RHI::ImageDescriptor imageDesc = AZ::RHI::ImageDescriptor::Create3D(
            AZ::RHI::ImageBindFlags::ShaderReadWrite, pixelSize, pixelSize, pixelSize, AZ::RHI::Format::R8G8B8A8_UNORM);
        imageDesc.m_mipLevels = CloudTextureComputePass::CalculateMipCount(pixelSize);
        AZ::RHI::ClearValue clearValue = AZ::RHI::ClearValue::CreateVector4Float(0, 0, 0, 0);
        AZ::Data::Instance<AZ::RPI::AttachmentImagePool> pool = AZ::RPI::ImageSystemInterface::Get()->GetSystemAttachmentPool();
        return AZ::RPI::AttachmentImage::Create(*pool.get(), imageDesc, AZ::Name("CloudTextureAttachmentImage"), &clearValue, nullptr);
    }


    AZStd::shared_ptr<CloudTextureComputePipeline> CloudTexturesComputeFeatureProcessor::CreateTextureComputeTask(
        AZStd::shared_ptr<CloudTextureComputeRequest> request)
    {
        request->m_cloudTextureAttachment = CreateTexture3DAttachmentImage(request->m_computeData.m_pixelSize);
        const bool validateMips = r_cloudsValidateNoiseMips;
        auto texture3DReadyCB = [this, request, validateMips](CloudTextureComputePipeline::RenderTaskId,
            const AZStd::vector<CloudTextureComputePipeline::CloudTextureSubresourceReadback>& results)
        {
            if (validateMips)
            {
                const bool valid = ValidateNoiseMips(results, request->m_computeData.m_pixelSize);
                AZ_Error(LogName, valid, "GPU noise mip validation FAILED for %u^3 texture", request->m_computeData.m_pixelSize);
                if (valid)
                {
                    AZ_Info(LogName, "GPU noise mip validation PASSED: %u^3, %zu levels", request->m_computeData.m_pixelSize, results.size());
                }
            }
            // Remove before signaling: listeners may enqueue another request or
            // deactivate themselves. The captured request stays alive throughout.
            bool current = false;
            for (auto it = m_computeRequests.begin(); it != m_computeRequests.end(); ++it)
            {
                if (it->second == request)
                {
                    current = true;
                    m_computeRequests.erase(it);
                    break;
                }
            }
            if (!current)
            {
                return;
            }
            // Cache only successful, still-observed work; a stale slider request
            // should not evict useful textures. Include all mip levels in the cap.
            size_t bytes = 0;
            for (uint32_t size = request->m_computeData.m_pixelSize; size; size >>= 1)
            {
                bytes += size_t(size) * size * size * 4;
            }
            if (bytes <= TextureCacheBudget)
            {
                while (!m_textureCache.empty() && m_cachedBytes + bytes > TextureCacheBudget)
                {
                    m_cachedBytes -= m_textureCache.front().m_bytes;
                    m_textureCache.pop_front();
                }
                m_textureCache.push_back({request->m_computeData, request->m_cloudTextureAttachment, bytes});
                m_cachedBytes += bytes;
            }
            request->m_readyEvent.Signal(request->m_cloudTextureAttachment);
            for (const auto& result : results)
            {
                request->m_readbackEvent.Signal(request->m_cloudTextureAttachment,
                    result.m_dataBuffer, result.m_mipSlice, result.m_mipSize);
            }
        };
        auto task = AZStd::make_shared<CloudTextureComputePipeline>();
        request->m_textureComputeTaskId = task->StartTextureCompute(m_computeScene.get(),
            request->m_cloudTextureAttachment, request->m_computeData,
            AZStd::move(texture3DReadyCB), request->m_withAttachmentReadback || validateMips);
        if (!request->m_textureComputeTaskId)
        {
            for (auto it = m_computeRequests.begin(); it != m_computeRequests.end(); ++it)
            {
                if (it->second == request)
                {
                    m_computeRequests.erase(it);
                    break;
                }
            }
        }
        return task;
    }
} // namespace VolumetricClouds
