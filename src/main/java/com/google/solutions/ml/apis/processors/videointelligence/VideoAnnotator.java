/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.solutions.ml.apis.processors.videointelligence;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import com.google.api.client.util.ExponentialBackOff;
import com.google.api.gax.rpc.BidiStream;
import com.google.api.gax.rpc.ResourceExhaustedException;
import com.google.cloud.videointelligence.v1p3beta1.*;
import com.google.protobuf.ByteString;
import com.google.protobuf.GeneratedMessageV3;
import com.google.solutions.ml.apis.BackOffUtils;
import com.google.solutions.ml.apis.GCSFileInfo;
import com.google.solutions.ml.apis.GCSUtils;
import com.google.solutions.ml.apis.VisionAnalyticsPipeline;
import org.apache.beam.sdk.values.KV;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VideoAnnotator {

    public static final Logger LOG = LoggerFactory.getLogger(VideoAnnotator.class);
    private final StreamingVideoConfig streamingVideoConfig;
    private final StreamingVideoIntelligenceServiceClient client;
    BidiStream<StreamingAnnotateVideoRequest, StreamingAnnotateVideoResponse> streamCall;

    public VideoAnnotator(StreamingFeature feature) throws IOException {
        StreamingObjectTrackingConfig objectTrackingConfig =
                StreamingObjectTrackingConfig.newBuilder().build();
        StreamingLabelDetectionConfig labelConfig =
                StreamingLabelDetectionConfig.newBuilder().build();
        StreamingVideoConfig.Builder builder = StreamingVideoConfig.newBuilder()
            .setObjectTrackingConfig(objectTrackingConfig)
            .setLabelDetectionConfig(labelConfig);
        if (feature != null) {
            builder = builder.setFeature(feature);
        }
        this.streamingVideoConfig = builder.build();
        this.client = StreamingVideoIntelligenceServiceClient.create();
    }

    public void teardown() {
        if (client != null) {
            client.shutdown();
            try {
                int waitTime = 10;
                if (!client.awaitTermination(waitTime, TimeUnit.SECONDS)) {
                    LOG.warn(
                            "Failed to shutdown the annotation client after {} seconds. Closing client anyway.",
                            waitTime);
                }
            } catch (InterruptedException e) {
                // Do nothing
            }
            client.close();
        }
    }

    public List<KV<GCSFileInfo, GeneratedMessageV3>> processFiles(Iterable<GCSFileInfo> fileURIs) {
        List<KV<GCSFileInfo, GeneratedMessageV3>> result = new ArrayList<>();
        ExponentialBackOff backoff = BackOffUtils.createBackOff();
        fileURIs.forEach(
            fileInfo -> {
                while (true) {
                    try {
                        // Download file's content from GCS
                        ByteString bytes = GCSUtils.getBytes(fileInfo.getUri());

                        // Send the bytes to the Streaming Video API
                        streamCall = client.streamingAnnotateVideoCallable().call();
                        streamCall.send(
                                StreamingAnnotateVideoRequest.newBuilder()
                                        .setVideoConfig(streamingVideoConfig)
                                        .build());
                        streamCall.send(StreamingAnnotateVideoRequest.newBuilder().setInputContent(bytes).build());
                        VisionAnalyticsPipeline.numberOfVideoApiRequests.inc();
                        streamCall.closeSend();
                        for (StreamingAnnotateVideoResponse response : streamCall) {
                            result.add(KV.of(fileInfo, response));
                        }
                        break;
                    } catch (ResourceExhaustedException e) {
                        BackOffUtils.handleQuotaReachedException(backoff, e);
                    }
                }
            });
        return result;
    }

}
