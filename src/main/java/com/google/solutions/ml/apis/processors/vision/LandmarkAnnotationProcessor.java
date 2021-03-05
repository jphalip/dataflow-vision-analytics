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
import com.google.cloud.vision.v1.EntityAnnotation;
import com.google.common.collect.ImmutableList;
import com.google.solutions.ml.apis.BQDestination;
import com.google.solutions.ml.apis.GCSFileInfo;
import com.google.solutions.ml.apis.TableDetails;
import com.google.solutions.ml.apis.TableSchemaProducer;
import com.google.solutions.ml.apis.processors.Constants;
import com.google.solutions.ml.apis.processors.MLApiResponseProcessor;
import com.google.solutions.ml.apis.processors.ProcessorUtils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import com.google.solutions.ml.apis.*;
import org.apache.beam.sdk.metrics.Counter;
import org.apache.beam.sdk.metrics.Metrics;
import org.apache.beam.sdk.values.KV;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Extracts landmark annotations (https://cloud.google.com/vision/docs/detecting-landmarks)
 */
public class LandmarkAnnotationProcessor extends ImageMLApiResponseProcessor {

  private static final long serialVersionUID = 1L;

  private static final Logger LOG = LoggerFactory.getLogger(LandmarkAnnotationProcessor.class);
  public final static Counter counter =
      Metrics.counter(MLApiResponseProcessor.class, "numberOfLandmarkAnnotations");

  private static class SchemaProducer implements TableSchemaProducer {

    private static final long serialVersionUID = 1L;

    @Override
    public TableSchema getTableSchema() {
      return new TableSchema().setFields(
          ImmutableList.of(
              new TableFieldSchema()
                  .setName(Constants.Field.GCS_URI_FIELD)
                  .setType(BigQueryConstants.Type.STRING)
                  .setMode(BigQueryConstants.Mode.REQUIRED),
              new TableFieldSchema()
                  .setName(Constants.Field.MID_FIELD).setType(BigQueryConstants.Type.STRING)
                  .setMode(BigQueryConstants.Mode.NULLABLE),
              new TableFieldSchema()
                  .setName(Constants.Field.DESCRIPTION_FIELD).setType(BigQueryConstants.Type.STRING)
                  .setMode(BigQueryConstants.Mode.REQUIRED),
              new TableFieldSchema()
                  .setName(Constants.Field.SCORE_FIELD).setType(BigQueryConstants.Type.FLOAT)
                  .setMode(BigQueryConstants.Mode.REQUIRED),
              new TableFieldSchema()
                  .setName(Constants.Field.BOUNDING_POLY).setType(BigQueryConstants.Type.RECORD)
                  .setMode(BigQueryConstants.Mode.NULLABLE).setFields(Constants.POLYGON_FIELDS),
              new TableFieldSchema()
                  .setName(Constants.Field.LOCATIONS).setType(BigQueryConstants.Type.GEOGRAPHY).setMode(BigQueryConstants.Mode.REPEATED),
              new TableFieldSchema()
                  .setName(Constants.Field.TIMESTAMP_FIELD).setType(BigQueryConstants.Type.TIMESTAMP)
                  .setMode(BigQueryConstants.Mode.REQUIRED))
      );
    }
  }

  @Override
  public TableDetails destinationTableDetails() {
    return TableDetails.create("Google Vision API Landmark Annotations",
        new Clustering().setFields(Collections.singletonList(Constants.Field.GCS_URI_FIELD)),
        new TimePartitioning().setField(Constants.Field.TIMESTAMP_FIELD), new SchemaProducer());
  }

  private final BQDestination destination;

  /**
   * Creates a processor and specifies the table id to persist to.
   */
  public LandmarkAnnotationProcessor(String tableId) {
    destination = new BQDestination(tableId);
  }

  @Override
  public Iterable<KV<BQDestination, TableRow>> process(GCSFileInfo fileInfo, GeneratedMessageV3 r) {
    AnnotateImageResponse response = (AnnotateImageResponse) r;
    int numberOfAnnotations = response.getLandmarkAnnotationsCount();
    if (numberOfAnnotations == 0) {
      return null;
    }

    counter.inc(numberOfAnnotations);

    Collection<KV<BQDestination, TableRow>> result = new ArrayList<>(numberOfAnnotations);
    for (EntityAnnotation annotation : response.getLandmarkAnnotationsList()) {
      TableRow row = ProcessorUtils.startRow(fileInfo);
      row.put(Constants.Field.MID_FIELD, annotation.getMid());
      row.put(Constants.Field.DESCRIPTION_FIELD, annotation.getDescription());
      row.put(Constants.Field.SCORE_FIELD, annotation.getScore());

      ProcessorUtils.extractBoundingPoly(annotation, row);

      if (annotation.getLocationsCount() > 0) {
        List<String> locations = new ArrayList<>(annotation.getLocationsCount());
        annotation.getLocationsList().forEach(
            location -> locations.add(
                "POINT(" + location.getLatLng().getLongitude() + " " +
                    location.getLatLng().getLatitude() + ")"));
        row.put(Constants.Field.LOCATIONS, locations);
      }

      LOG.debug("Processing {}", row);
      result.add(KV.of(destination, row));
    }

    return result;
  }

}
