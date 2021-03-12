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

import com.google.protobuf.GeneratedMessageV3;
import com.google.api.services.bigquery.model.Clustering;
import com.google.api.services.bigquery.model.TableFieldSchema;
import com.google.api.services.bigquery.model.TableRow;
import com.google.api.services.bigquery.model.TableSchema;
import com.google.api.services.bigquery.model.TimePartitioning;
import com.google.cloud.vision.v1.AnnotateImageResponse;
import com.google.common.collect.ImmutableList;
import com.google.solutions.ml.apis.BQDestination;
import com.google.solutions.ml.apis.GCSFileInfo;
import com.google.solutions.ml.apis.TableDetails;
import com.google.solutions.ml.apis.TableSchemaProducer;
import com.google.solutions.ml.apis.processors.MLApiResponseProcessor;
import com.google.solutions.ml.apis.processors.ProcessorUtils;
import com.google.solutions.ml.apis.processors.Constants.Field;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Set;

import com.google.solutions.ml.apis.*;
import org.apache.beam.sdk.metrics.Counter;
import org.apache.beam.sdk.metrics.Metrics;
import org.apache.beam.sdk.values.KV;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Captures the error occurred during processing. Note, that there could be some valid annotations
 * returned in the response even though the response contains an error.
 */
public class ErrorProcessor extends ImageMLApiResponseProcessor {

  private static final long serialVersionUID = 1L;

  public final static Counter counter =
      Metrics.counter(MLApiResponseProcessor.class, "numberOfErrors");
  public static final Logger LOG = LoggerFactory.getLogger(ErrorProcessor.class);

  private final BQDestination destination;
  private final Set<String> metadataKeys;

  /**
   * Creates a processor and specifies the table id to persist to.
   */
  public ErrorProcessor(String tableId, Set<String> metadataKeys) {
    this.destination = new BQDestination(tableId);
    this.metadataKeys = metadataKeys;
  }

  private static class SchemaProducer implements TableSchemaProducer {

    private static final long serialVersionUID = 1L;
    private final Set<String> metadataKeys;

    SchemaProducer(Set<String> metadataKeys) {
      this.metadataKeys = metadataKeys;
    }

    @Override
    public TableSchema getTableSchema() {
      ArrayList<TableFieldSchema> fields = new ArrayList<>();
      fields.add(
          new TableFieldSchema()
              .setName(Field.GCS_URI_FIELD)
              .setType(BigQueryConstants.Type.STRING)
              .setMode(BigQueryConstants.Mode.REQUIRED));
      fields.add(
          new TableFieldSchema()
              .setName(Field.DESCRIPTION_FIELD).setType(BigQueryConstants.Type.STRING)
              .setMode(BigQueryConstants.Mode.REQUIRED));
      fields.add(
          new TableFieldSchema()
              .setName(Field.STACK_TRACE).setType(BigQueryConstants.Type.STRING)
              .setMode(BigQueryConstants.Mode.NULLABLE));
      fields.add(
          new TableFieldSchema()
              .setName(Field.TIMESTAMP_FIELD).setType(BigQueryConstants.Type.TIMESTAMP)
              .setMode(BigQueryConstants.Mode.REQUIRED));
      ProcessorUtils.setMetadataFieldsSchema(fields, metadataKeys);
      return new TableSchema().setFields(fields);
    }
  }

  @Override
  public TableDetails destinationTableDetails() {
    return TableDetails.create("Google Vision API Processing Errors",
        new Clustering().setFields(Collections.singletonList(Field.GCS_URI_FIELD)),
        new TimePartitioning().setField(Field.TIMESTAMP_FIELD), new SchemaProducer(metadataKeys));
  }

  @Override
  public Iterable<KV<BQDestination, TableRow>> process(GCSFileInfo fileInfo, GeneratedMessageV3 r) {
    AnnotateImageResponse response = (AnnotateImageResponse) r;
    if (!response.hasError()) {
      return null;
    }

    counter.inc();

    TableRow row = ProcessorUtils.startRow(fileInfo);
    row.put(Field.DESCRIPTION_FIELD, response.getError().toString());
    ProcessorUtils.addMetadataValues(row, fileInfo, metadataKeys);

    LOG.debug("Processing {}", row);

    return Collections.singletonList(KV.of(destination, row));
  }
}
