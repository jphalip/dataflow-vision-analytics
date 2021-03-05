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

import com.google.api.services.bigquery.model.TableRow;
import com.google.auto.value.AutoValue;
import java.util.Collection;

import com.google.protobuf.GeneratedMessageV3;
import com.google.solutions.ml.apis.processors.MLApiResponseProcessor;
import org.apache.beam.sdk.metrics.Counter;
import org.apache.beam.sdk.metrics.Metrics;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.values.KV;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ProcessImageResponse {@link ProcessMLApiResponseDoFn} class parses the API response for
 * specific annotation and using response builder output the table and table row for BigQuery
 */
@AutoValue
abstract public class ProcessMLApiResponseDoFn
    extends DoFn<KV<GCSFileInfo, GeneratedMessageV3>, KV<BQDestination, TableRow>> {

  private static final long serialVersionUID = 1L;
  private static final Logger LOG = LoggerFactory.getLogger(ProcessMLApiResponseDoFn.class);

  abstract Collection<MLApiResponseProcessor> processors();

  abstract Counter processedFileCounter();

  public static ProcessMLApiResponseDoFn create(
      Collection<MLApiResponseProcessor> processors) {
    return builder()
        .processors(processors)
        .processedFileCounter(Metrics
            .counter(ProcessMLApiResponseDoFn.class, "processedFiles"))
        .build();
  }

  @ProcessElement
  public void processElement(@Element KV<GCSFileInfo, GeneratedMessageV3> element,
      OutputReceiver<KV<BQDestination, TableRow>> out) {
    GCSFileInfo fileInfo = element.getKey();
    GeneratedMessageV3 annotationResponse = element.getValue();

    LOG.debug("Processing annotations for file: {}", fileInfo.getUri());
    processedFileCounter().inc();

    processors().forEach(processor -> {
      if (processor.shouldProcess(annotationResponse)) {
        Iterable<KV<BQDestination, TableRow>> processingResult = processor
                .process(fileInfo, annotationResponse);
        if (processingResult != null) {
          processingResult.forEach(out::output);
        }
      }
    });
  }

  public static Builder builder() {
    return new AutoValue_ProcessMLApiResponseDoFn.Builder();
  }


  @AutoValue.Builder
  public abstract static class Builder {

    public abstract Builder processors(Collection<MLApiResponseProcessor> processors);

    public abstract Builder processedFileCounter(Counter processedFileCounter);

    public abstract ProcessMLApiResponseDoFn build();
  }
}
