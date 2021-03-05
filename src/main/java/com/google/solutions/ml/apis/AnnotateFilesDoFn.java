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

package com.google.solutions.ml.apis;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.google.cloud.videointelligence.v1p3beta1.StreamingFeature;
import com.google.cloud.vision.v1.Feature;
import com.google.protobuf.GeneratedMessageV3;
import com.google.solutions.ml.apis.processors.videointelligence.VideoAnnotator;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.values.KV;

public class AnnotateFilesDoFn extends DoFn<Iterable<GCSFileInfo>, KV<GCSFileInfo, GeneratedMessageV3>> {

    private static final long serialVersionUID = 1L;
    private final StreamingFeature videoFeature;
    private VideoAnnotator videoAnnotator;
    private ImageAnnotator imageAnnotator;
    private final List<Feature.Type> imageFeatures;

    public AnnotateFilesDoFn(List<Feature.Type> imageFeatures, StreamingFeature videoFeature) {
        this.imageFeatures = imageFeatures;
        this.videoFeature = videoFeature;
    }

    @Setup
    public void setup() throws IOException {
        imageAnnotator = new ImageAnnotator(imageFeatures);
        videoAnnotator = new VideoAnnotator(videoFeature);
    }

    @Teardown
    public void teardown() {
        imageAnnotator.teardown();
        videoAnnotator.teardown();
    }

    @ProcessElement
    public void processElement(@Element Iterable<GCSFileInfo> fileInfos,
                               OutputReceiver<KV<GCSFileInfo, GeneratedMessageV3>> out) {
        List<GCSFileInfo> videoFiles = new ArrayList<>();
        List<GCSFileInfo> imageFiles = new ArrayList<>();
        for (GCSFileInfo fileInfo: fileInfos) {
            if (VisionAnalyticsPipeline.SUPPORTED_IMAGE_CONTENT_TYPES.contains(fileInfo.getContentType())) {
                imageFiles.add(fileInfo);
            }
            else {
                videoFiles.add(fileInfo);
            }
        }

        List<KV<GCSFileInfo, GeneratedMessageV3>> responses = new ArrayList<>();
        if (! imageFiles.isEmpty()) {
            responses.addAll(imageAnnotator.processFiles(imageFiles));
        }
        if (! videoFiles.isEmpty()) {
            responses.addAll(videoAnnotator.processFiles(videoFiles));
        }

        int index = 0;
        for (KV<GCSFileInfo, GeneratedMessageV3> response : responses) {
            out.output(response);
        }
    }
}
