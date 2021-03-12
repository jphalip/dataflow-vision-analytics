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
import com.google.cloud.vision.v1.FaceAnnotation;
import com.google.cloud.vision.v1.Position;
import com.google.common.collect.ImmutableList;
import com.google.solutions.ml.apis.BQDestination;
import com.google.solutions.ml.apis.GCSFileInfo;
import com.google.solutions.ml.apis.TableDetails;
import com.google.solutions.ml.apis.TableSchemaProducer;
import com.google.solutions.ml.apis.processors.Constants;
import com.google.solutions.ml.apis.processors.MLApiResponseProcessor;
import com.google.solutions.ml.apis.processors.ProcessorUtils;
import com.google.solutions.ml.apis.processors.Constants.Field;

import java.util.*;

import com.google.solutions.ml.apis.BigQueryConstants;
import org.apache.beam.sdk.metrics.Counter;
import org.apache.beam.sdk.metrics.Metrics;
import org.apache.beam.sdk.values.KV;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Extracts face annotations (https://cloud.google.com/vision/docs/detecting-faces)
 */
public class FaceAnnotationProcessor extends ImageMLApiResponseProcessor {

  private static final long serialVersionUID = 1L;

  private static final Logger LOG = LoggerFactory.getLogger(FaceAnnotationProcessor.class);
  public final static Counter counter =
      Metrics.counter(MLApiResponseProcessor.class, "numberOfFaceAnnotations");
  private final BQDestination destination;
  private final Set<String> metadataKeys;

  /**
   * Creates a processor and specifies the table id to persist to.
   */
  public FaceAnnotationProcessor(String tableId, Set<String> metadataKeys) {
      this.destination = new BQDestination(tableId);
      this.metadataKeys = metadataKeys;
  }

  /**
   * The schema doesn't represent the complete list of all attributes returned by the APIs. For more
   * details see https://cloud.google.com/vision/docs/reference/rest/v1/AnnotateImageResponse?hl=pl#FaceAnnotation
   */
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
              .setName(Field.GCS_URI_FIELD).setType(BigQueryConstants.Type.STRING).setMode(BigQueryConstants.Mode.REQUIRED));
      fields.add(
          new TableFieldSchema()
              .setName(Field.BOUNDING_POLY).setType(BigQueryConstants.Type.RECORD)
              .setMode(BigQueryConstants.Mode.REQUIRED).setFields(Constants.POLYGON_FIELDS));
      fields.add(
          new TableFieldSchema()
              .setName(Field.FD_BOUNDING_POLY).setType(BigQueryConstants.Type.RECORD)
              .setMode(BigQueryConstants.Mode.REQUIRED).setFields(Constants.POLYGON_FIELDS));
      fields.add(
          new TableFieldSchema()
              .setName(Field.LANDMARKS).setType(BigQueryConstants.Type.RECORD).setMode(BigQueryConstants.Mode.REPEATED).setFields(
              Arrays.asList(
                  new TableFieldSchema().setName(Field.FACE_LANDMARK_TYPE).setType(BigQueryConstants.Type.STRING)
                      .setMode(BigQueryConstants.Mode.REQUIRED),
                  new TableFieldSchema().setName(Field.FACE_LANDMARK_POSITION).setType(BigQueryConstants.Type.RECORD)
                      .setMode(BigQueryConstants.Mode.REQUIRED).setFields(Constants.POSITION_FIELDS)
              )
          ));
      fields.add(
          new TableFieldSchema()
              .setName(Field.DETECTION_CONFIDENCE).setType(BigQueryConstants.Type.FLOAT).setMode(BigQueryConstants.Mode.REQUIRED));
      fields.add(
          new TableFieldSchema()
              .setName(Field.LANDMARKING_CONFIDENCE).setType(BigQueryConstants.Type.FLOAT).setMode(BigQueryConstants.Mode.REQUIRED));
      fields.add(
          new TableFieldSchema()
              .setName(Field.JOY_LIKELIHOOD).setType(BigQueryConstants.Type.STRING).setMode(BigQueryConstants.Mode.REQUIRED));
      fields.add(
          new TableFieldSchema()
              .setName(Field.SORROW_LIKELIHOOD).setType(BigQueryConstants.Type.STRING).setMode(BigQueryConstants.Mode.REQUIRED));
      fields.add(
          new TableFieldSchema()
              .setName(Field.ANGER_LIKELIHOOD).setType(BigQueryConstants.Type.STRING).setMode(BigQueryConstants.Mode.REQUIRED));
      fields.add(
          new TableFieldSchema()
              .setName(Field.SURPRISE_LIKELIHOOD).setType(BigQueryConstants.Type.STRING).setMode(BigQueryConstants.Mode.REQUIRED));
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
    return TableDetails.create("Google Vision API Face Annotations",
        new Clustering().setFields(Collections.singletonList(Field.GCS_URI_FIELD)),
        new TimePartitioning().setField(Field.TIMESTAMP_FIELD), new SchemaProducer(metadataKeys));
  }

  @Override
  public Iterable<KV<BQDestination, TableRow>> process(GCSFileInfo fileInfo, GeneratedMessageV3 r) {
      AnnotateImageResponse response = (AnnotateImageResponse) r;
    int numberOfAnnotations = response.getFaceAnnotationsCount();
    if (numberOfAnnotations == 0) {
      return null;
    }

    counter.inc(numberOfAnnotations);

    Collection<KV<BQDestination, TableRow>> result = new ArrayList<>(numberOfAnnotations);
    for (FaceAnnotation annotation : response.getFaceAnnotationsList()) {
      TableRow row = ProcessorUtils.startRow(fileInfo);

      row.put(Field.BOUNDING_POLY,
          ProcessorUtils.getBoundingPolyAsRow(annotation.getBoundingPoly()));
      row.put(Field.FD_BOUNDING_POLY,
          ProcessorUtils.getBoundingPolyAsRow(annotation.getFdBoundingPoly()));
      List<TableRow> landmarks = new ArrayList<>(annotation.getLandmarksCount());
      annotation.getLandmarksList().forEach(
          landmark -> {
            TableRow landmarkRow = new TableRow();
            landmarkRow.put(Field.FACE_LANDMARK_TYPE, landmark.getType().toString());

            Position position = landmark.getPosition();
            TableRow positionRow = new TableRow();
            positionRow.put(Field.VERTEX_X, position.getX());
            positionRow.put(Field.VERTEX_Y, position.getY());
            positionRow.put(Field.VERTEX_Z, position.getZ());
            landmarkRow.put(Field.FACE_LANDMARK_POSITION, positionRow);

            landmarks.add(landmarkRow);
          }
      );
      row.put(Field.LANDMARKS, landmarks);
      row.put(Field.DETECTION_CONFIDENCE, annotation.getDetectionConfidence());
      row.put(Field.LANDMARKING_CONFIDENCE, annotation.getLandmarkingConfidence());
      row.put(Field.JOY_LIKELIHOOD, annotation.getJoyLikelihood().toString());
      row.put(Field.SORROW_LIKELIHOOD, annotation.getSorrowLikelihood().toString());
      row.put(Field.ANGER_LIKELIHOOD, annotation.getAngerLikelihood().toString());
      row.put(Field.SURPRISE_LIKELIHOOD, annotation.getSurpriseLikelihood().toString());
      ProcessorUtils.addMetadataValues(row, fileInfo, metadataKeys);

      LOG.debug("Processing {}", row);
      result.add(KV.of(destination, row));
    }

    return result;
  }

}
