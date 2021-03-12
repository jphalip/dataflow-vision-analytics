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

import java.util.*;

import com.google.api.services.bigquery.model.*;
import com.google.cloud.videointelligence.v1p3beta1.NormalizedBoundingBox;
import com.google.cloud.videointelligence.v1p3beta1.ObjectTrackingAnnotation;
import com.google.cloud.videointelligence.v1p3beta1.StreamingAnnotateVideoResponse;
import com.google.cloud.videointelligence.v1p3beta1.StreamingVideoAnnotationResults;
import com.google.common.collect.ImmutableList;
import com.google.protobuf.GeneratedMessageV3;
import com.google.solutions.ml.apis.BQDestination;
import com.google.solutions.ml.apis.GCSFileInfo;
import com.google.solutions.ml.apis.TableDetails;
import com.google.solutions.ml.apis.TableSchemaProducer;
import com.google.solutions.ml.apis.processors.MLApiResponseProcessor;
import com.google.solutions.ml.apis.processors.ProcessorUtils;
import com.google.solutions.ml.apis.processors.Constants.Field;
import org.apache.beam.sdk.metrics.Counter;
import org.apache.beam.sdk.metrics.Metrics;
import org.apache.beam.sdk.values.KV;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VideoObjectTrackingAnnotationProcessor extends VideoMLApiResponseProcessor {

    private static final long serialVersionUID = 1L;

    private static final Logger LOG = LoggerFactory.getLogger(VideoObjectTrackingAnnotationProcessor.class);
    private final BQDestination destination;
    private final Set<String> metadataKeys;
    public final static Counter counter =
            Metrics.counter(MLApiResponseProcessor.class, "numberOfVideoObjectTrackingAnnotations");

    /**
     * Creates a processor and specifies the table id to persist to.
     */
    public VideoObjectTrackingAnnotationProcessor(String tableId, Set<String> metadataKeys) {
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
                    .setType("STRING")
                    .setMode("REQUIRED"));
            fields.add(
                new TableFieldSchema()
                    .setName(Field.TIMESTAMP_FIELD)
                    .setType("TIMESTAMP")
                    .setMode("REQUIRED"));
            fields.add(
                new TableFieldSchema()
                    .setName(Field.ENTITY)
                    .setType("STRING")
                    .setMode("REQUIRED"));
            fields.add(
                new TableFieldSchema()
                    .setName(Field.CONFIDENCE).setType("FLOAT"));
            fields.add(
                new TableFieldSchema()
                    .setName(Field.FRAMES).setType("RECORD")
                    .setMode("REPEATED")
                    .setFields(ImmutableList.of(
                            new TableFieldSchema()
                                    .setName(Field.TIME_OFFSET)
                                    .setType("STRING")
                                    .setMode("REQUIRED"),
                            new TableFieldSchema()
                                    .setName(Field.LEFT)
                                    .setType("FLOAT")
                                    .setMode("REQUIRED"),
                            new TableFieldSchema()
                                    .setName(Field.TOP)
                                    .setType("FLOAT")
                                    .setMode("REQUIRED"),
                            new TableFieldSchema()
                                    .setName(Field.RIGHT)
                                    .setType("FLOAT")
                                    .setMode("REQUIRED"),
                            new TableFieldSchema()
                                    .setName(Field.BOTTOM)
                                    .setType("FLOAT")
                                    .setMode("REQUIRED")
                    ))
                );
            ProcessorUtils.setMetadataFieldsSchema(fields, metadataKeys);
            return new TableSchema().setFields(fields);
        }

    }

    @Override
    public Iterable<KV<BQDestination, TableRow>> process(GCSFileInfo fileInfo, GeneratedMessageV3 r) {
        StreamingAnnotateVideoResponse response = (StreamingAnnotateVideoResponse) r;
        StreamingVideoAnnotationResults annotationResults = response.getAnnotationResults();
        int numberOfAnnotations = annotationResults.getObjectAnnotationsCount();
        if (numberOfAnnotations == 0) {
            return null;
        }
        counter.inc(numberOfAnnotations);
        Collection<KV<BQDestination, TableRow>> result = new ArrayList<>(numberOfAnnotations);
        for (ObjectTrackingAnnotation annotation : annotationResults.getObjectAnnotationsList()) {
            TableRow row = ProcessorUtils.startRow(fileInfo);
            row.set(Field.CONFIDENCE, annotation.getConfidence());
            row.set(Field.ENTITY, annotation.hasEntity() ? annotation.getEntity().getDescription() : "NOT_FOUND");
            List<TableRow> frames = new ArrayList<>(annotation.getFramesCount());
            annotation
                .getFramesList()
                .forEach(
                    frame -> {
                        TableRow frameRow = new TableRow();
                        NormalizedBoundingBox normalizedBoundingBox = frame.getNormalizedBoundingBox();
                        frameRow.set(Field.TIME_OFFSET, frame.getTimeOffset());
                        frameRow.set(Field.LEFT, normalizedBoundingBox.getLeft());
                        frameRow.set(Field.TOP, normalizedBoundingBox.getTop());
                        frameRow.set(Field.RIGHT, normalizedBoundingBox.getRight());
                        frameRow.set(Field.BOTTOM, normalizedBoundingBox.getBottom());
                        frames.add(frameRow);
                    });
            row.put(Field.FRAMES, frames);
            ProcessorUtils.addMetadataValues(row, fileInfo, metadataKeys);
            LOG.debug("Processing {}", row);
            result.add(KV.of(destination, row));
        }
        return result;
    }

    @Override
    public TableDetails destinationTableDetails() {
        return TableDetails.create("Google Video Intelligence API object tracking annotations",
                new Clustering().setFields(Collections.singletonList(Field.GCS_URI_FIELD)),
                new TimePartitioning().setField(Field.TIMESTAMP_FIELD), new SchemaProducer(metadataKeys));
    }
}
