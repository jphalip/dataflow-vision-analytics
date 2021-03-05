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

package com.google.solutions.ml.apis.processors.vision;

import com.google.cloud.vision.v1.AnnotateImageResponse;
import com.google.protobuf.GeneratedMessageV3;
import com.google.solutions.ml.apis.processors.MLApiResponseProcessor;


public abstract class ImageMLApiResponseProcessor implements MLApiResponseProcessor {

    private static final long serialVersionUID = 1L;

    @Override
    public boolean shouldProcess(GeneratedMessageV3 response) {
        return (response instanceof AnnotateImageResponse);
    }

}
