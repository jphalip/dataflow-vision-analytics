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

package com.google.solutions.ml.apis.processors;

import java.io.Serializable;

import com.google.protobuf.GeneratedMessageV3;
import com.google.api.services.bigquery.model.TableRow;
import com.google.solutions.ml.apis.BQDestination;
import com.google.solutions.ml.apis.GCSFileInfo;
import com.google.solutions.ml.apis.TableDetails;
import org.apache.beam.sdk.values.KV;

/**
 * Implementors of this interface will process zero to many TableRows to persist to a specific
 * BigTable table.
 */
public interface MLApiResponseProcessor extends Serializable {

    /**
     * @param fileInfo annotation source
     * @param response from Google Cloud ML Video Intelligence or Cloud ML Vision API
     * @return key/value pair of a BigQuery destination and a TableRow to persist.
     */
    Iterable<KV<BQDestination, TableRow>> process(GCSFileInfo fileInfo, GeneratedMessageV3 response);

    /**
     * @return details of the table to persist to.
     */
    TableDetails destinationTableDetails();

    /**
     * @return true if the processor is meant to processor this type of response object.
     */
    boolean shouldProcess(GeneratedMessageV3 response);
}