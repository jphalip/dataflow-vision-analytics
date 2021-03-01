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

package com.google.solutions.ml.api.vision.videointelligence;

public interface Constants {

    interface Field {
        String GCS_URI_FIELD = "gcs_uri";
        String TIMESTAMP_FIELD = "transaction_timestamp";
        String METADATA = "metadata";
        String ENTITY = "entity";
        String SEGMENTS = "segments";
        String START_TIME_OFFSET = "start_time_offset";
        String END_TIME_OFFSET = "end_time_offset";
        String FRAMES = "frames";  // TODO: Rename to "frame_data"?
        String TIME_OFFSET = "time_offset";
        String CONFIDENCE = "confidence";
        String LEFT = "left";
        String TOP = "top";
        String RIGHT = "right";
        String BOTTOM = "bottom";
    }

}
