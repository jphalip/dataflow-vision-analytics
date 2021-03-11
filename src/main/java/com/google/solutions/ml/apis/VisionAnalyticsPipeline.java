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


import com.google.api.services.bigquery.model.TableFieldSchema;
import com.google.api.services.bigquery.model.TableReference;
import com.google.api.services.bigquery.model.TableRow;
import com.google.api.services.bigquery.model.TableSchema;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.protobuf.GeneratedMessageV3;
import com.google.solutions.ml.apis.BigQueryConstants.Mode;
import com.google.solutions.ml.apis.BigQueryConstants.Type;
import com.google.solutions.ml.apis.processors.MLApiResponseProcessor;
import com.google.solutions.ml.apis.processors.videointelligence.VideoLabelAnnotationProcessor;
import com.google.solutions.ml.apis.processors.videointelligence.VideoObjectTrackingAnnotationProcessor;
import com.google.solutions.ml.apis.processors.vision.CropHintAnnotationProcessor;
import com.google.solutions.ml.apis.processors.vision.ErrorProcessor;
import com.google.solutions.ml.apis.processors.vision.FaceAnnotationProcessor;
import com.google.solutions.ml.apis.processors.vision.ImagePropertiesProcessor;
import com.google.solutions.ml.apis.processors.vision.LabelAnnotationProcessor;
import com.google.solutions.ml.apis.processors.vision.LandmarkAnnotationProcessor;
import com.google.solutions.ml.apis.processors.vision.LogoAnnotationProcessor;
import com.google.solutions.ml.apis.processors.ProcessorUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.PipelineResult;
import org.apache.beam.sdk.io.FileIO;
import org.apache.beam.sdk.io.fs.MatchResult.Metadata;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO.Write.WriteDisposition;
import org.apache.beam.sdk.io.gcp.pubsub.PubsubIO;
import org.apache.beam.sdk.io.gcp.pubsub.PubsubMessage;
import org.apache.beam.sdk.metrics.Counter;
import org.apache.beam.sdk.metrics.Distribution;
import org.apache.beam.sdk.metrics.Metrics;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.transforms.Create;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.Filter;
import org.apache.beam.sdk.transforms.ParDo;
import org.apache.beam.sdk.transforms.SerializableFunction;
import org.apache.beam.sdk.transforms.windowing.AfterWatermark;
import org.apache.beam.sdk.transforms.windowing.BoundedWindow;
import org.apache.beam.sdk.transforms.windowing.FixedWindows;
import org.apache.beam.sdk.transforms.windowing.Window;
import org.apache.beam.sdk.values.KV;
import org.apache.beam.sdk.values.PCollection;
import org.joda.time.Duration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main class for the vision analytics processing.
 */
public class VisionAnalyticsPipeline {

  public static final Logger LOG = LoggerFactory.getLogger(VisionAnalyticsPipeline.class);

  public static final Counter totalFiles = Metrics
      .counter(VisionAnalyticsPipeline.class, "totalFiles");
  public static final Counter rejectedFiles = Metrics
      .counter(VisionAnalyticsPipeline.class, "rejectedFiles");
  public static final Counter numberOfImageApiRequests = Metrics
      .counter(VisionAnalyticsPipeline.class, "numberOfImageApiRequests");
  public static final Counter numberOfVideoApiRequests = Metrics
          .counter(VisionAnalyticsPipeline.class, "numberOfVideoApiRequests");
  public static final Counter numberOfQuotaExceededRequests = Metrics
      .counter(VisionAnalyticsPipeline.class, "numberOfQuotaExceededRequests");

  public static final Distribution batchSizeDistribution = Metrics
      .distribution(VisionAnalyticsPipeline.class, "batchSizeDistribution");

  public static final Set<String> SUPPORTED_IMAGE_CONTENT_TYPES = ImmutableSet.of(
      "image/jpeg", "image/png", "image/tiff", "image/tif", "image/gif"
  );

  public static final Set<String> SUPPORTED_VIDEO_CONTENT_TYPES = ImmutableSet.of(
      "video/mov", "video/mpeg4", "video/mp4", "video/avi"
  );

  public static final Set<String> SUPPORTED_CONTENT_TYPES =
      Sets.union(SUPPORTED_IMAGE_CONTENT_TYPES, SUPPORTED_VIDEO_CONTENT_TYPES).immutableCopy();

  public static final String ACCEPTED_IMAGE_FILE_PATTERN = "JPEG|jpeg|JPG|jpg|PNG|png|GIF|gif|TIFF|tiff|TIF|tif";

  public static final String ACCEPTED_VIDEO_FILE_PATTERN = "MP4|mp4|MOV|mov|MPEG4|mpeg4|AVI|avi";

  public static final String ACCEPTED_FILE_PATTERN = ACCEPTED_IMAGE_FILE_PATTERN + "|" + ACCEPTED_VIDEO_FILE_PATTERN;

  /**
   * Main entry point for executing the pipeline. This will run the pipeline asynchronously. If
   * blocking execution is required, use the {@link VisionAnalyticsPipeline#run(VisionAnalyticsPipelineOptions)}
   * method to start the pipeline and invoke {@code result.waitUntilFinish()} on the {@link
   * PipelineResult}
   *
   * @param args The command-line arguments to the pipeline.
   */
  public static void main(String[] args) throws IOException {

    VisionAnalyticsPipelineOptions options =
        PipelineOptionsFactory.fromArgs(args)
            .withValidation()
            .as(VisionAnalyticsPipelineOptions.class);

    run(options);
  }

  /**
   * Runs the pipeline
   *
   * @return result
   */
  public static PipelineResult run(VisionAnalyticsPipelineOptions options) throws IOException {
    Pipeline p = Pipeline.create(options);

    PCollection<GCSFileInfo> fileInfos;
    if (options.getInputNotificationSubscription() != null) {
      fileInfos = convertPubSubNotificationsToGCSFileInfos(p, options);
    } else if (options.getFileList() != null) {
      fileInfos = listGCSFiles(p, options);
    } else {
      throw new RuntimeException("Either the subscriber id or the file list should be provided.");
    }

    PCollection<Iterable<GCSFileInfo>> batchedFileInfos = fileInfos
        .apply("Batch files",
            BatchRequestsTransform.create(options.getBatchSize(), options.getKeyRange()));

    PCollection<KV<GCSFileInfo, GeneratedMessageV3>> annotatedFiles =
        options.isSimulate() ?
            batchedFileInfos.apply("Simulate Annotation",
                ParDo.of(new AnnotateFilesSimulatorDoFn(options.getImageFeatures(), options.getVideoFeature()))) :
            batchedFileInfos.apply(
                "Annotate files",
                ParDo.of(new AnnotateFilesDoFn(options.getImageFeatures(), options.getVideoFeature())));

    Map<String, MLApiResponseProcessor> processors = configureProcessors(options);

    PCollection<KV<BQDestination, TableRow>> annotationOutcome =
        annotatedFiles.apply(
            "Process Annotations",
            ParDo.of(ProcessMLApiResponseDoFn.create(ImmutableSet.copyOf(processors.values()))));

    annotationOutcome.apply("Write To BigQuery", new BigQueryDynamicWriteTransform(
        BQDynamicDestinations.builder()
            .project(options.getProject())
            .datasetName(options.getDatasetName())
            .tableNameToTableDetailsMap(
                tableNameToTableDetailsMap(processors)).build())
    );

    collectBatchStatistics(batchedFileInfos, options);

    return p.run();
  }

  /**
   * Collect the statistics on batching the requests. The results are published to a metric. If
   * {@link VisionAnalyticsPipelineOptions#isCollectBatchData()} is true the batch data is saved to
   * BigQuery table "batch_info".
   */
  static void collectBatchStatistics(PCollection<Iterable<GCSFileInfo>> batchedFileInfos,
      VisionAnalyticsPipelineOptions options) {

    PCollection<TableRow> batchInfo = batchedFileInfos
        .apply("Collect Batch Stats", ParDo.of(new DoFn<Iterable<GCSFileInfo>, TableRow>() {
          private static final long serialVersionUID = 1L;

          @ProcessElement
          public void processElement(@Element Iterable<GCSFileInfo> fileInfos, BoundedWindow window,
              OutputReceiver<TableRow> out, ProcessContext context) {
            int size = Iterables.size(fileInfos);
            batchSizeDistribution.update(size);
            if (context.getPipelineOptions().as(VisionAnalyticsPipelineOptions.class)
                .isCollectBatchData()) {
              TableRow row = new TableRow();
              row.put("window", window.toString());
              row.put("timestamp", ProcessorUtils.getTimeStamp());
              row.put("size", size);
              List<String> fileUris = new ArrayList<>();
              fileInfos.forEach((fileInfo) -> {
                fileUris.add(fileInfo.getUri());
              });
              row.put("items", fileUris);
              out.output(row);
            }
          }
        }));
    if (!options.isCollectBatchData()) {
      return;
    }
    batchInfo.apply(
        BigQueryIO.writeTableRows()
            .to(new TableReference().setProjectId(options.getProject())
                .setDatasetId(options.getDatasetName()).setTableId("batch_info"))
            .withWriteDisposition(WriteDisposition.WRITE_APPEND)
            .withoutValidation()
            .withClustering()
            .ignoreInsertIds()
            .withSchema(new TableSchema().setFields(ImmutableList.of(
                new TableFieldSchema().setName("window").setType(Type.STRING),
                new TableFieldSchema().setName("timestamp").setType(Type.TIMESTAMP),
                new TableFieldSchema().setName("size").setType(Type.NUMERIC),
                new TableFieldSchema().setName("items").setType(Type.STRING).setMode(Mode.REPEATED)
            )))
            .withCreateDisposition(BigQueryIO.Write.CreateDisposition.CREATE_IF_NEEDED));
  }

  /**
   * Create a map of the table details. Each processor will produce <code>TableRow</code>s destined
   * to a different table. Each processor will provide the details about that table.
   *
   * @return map of table details keyed by table name
   */
  static Map<String, TableDetails> tableNameToTableDetailsMap(
      Map<String, MLApiResponseProcessor> processors) {
    Map<String, TableDetails> tableNameToTableDetailsMap = new HashMap<>();
    processors.forEach(
        (tableName, processor) -> tableNameToTableDetailsMap
            .put(tableName, processor.destinationTableDetails()));
    return tableNameToTableDetailsMap;
  }

  /**
   * Reads PubSub messages from the subscription provided by {@link VisionAnalyticsPipelineOptions#getSubscriberId()}.
   *
   * The messages are expected to confirm to the GCS notification message format defined in
   * https://cloud.google.com/storage/docs/pubsub-notifications
   *
   * Notifications are filtered to have one of the supported content types: {@link
   * VisionAnalyticsPipeline#SUPPORTED_CONTENT_TYPES}.
   *
   * @return PCollection of GCS URIs
   */
  static PCollection<GCSFileInfo> convertPubSubNotificationsToGCSFileInfos(
      Pipeline p, VisionAnalyticsPipelineOptions options) {
    PCollection<GCSFileInfo> gcsFileInfos;
    PCollection<PubsubMessage> pubSubNotifications = p.begin().apply("Read PubSub",
        PubsubIO.readMessagesWithAttributes().fromSubscription(options.getInputNotificationSubscription()));
    gcsFileInfos = pubSubNotifications
        .apply("PubSub to GCS URIs",
            ParDo.of(PubSubNotificationToGCSInfoDoFn.create(SUPPORTED_CONTENT_TYPES)))
        .apply(
            "Fixed Window",
            Window.<GCSFileInfo>into(
                FixedWindows.of(Duration.standardSeconds(options.getWindowInterval())))
                .triggering(AfterWatermark.pastEndOfWindow())
                .discardingFiredPanes()
                .withAllowedLateness(Duration.standardMinutes(15)));
    return gcsFileInfos;
  }

  /**
   * Reads the GCS objects provided by {@link VisionAnalyticsPipelineOptions#getFileList()}.
   *
   * The file list can contain multiple entries. Each entry can contain wildcards supported by
   * {@link FileIO#matchAll()}.
   *
   * Files are filtered based on their suffixes as defined in {@link VisionAnalyticsPipeline#ACCEPTED_FILE_PATTERN}.
   *
   * @return PCollection of GCS URIs
   */
  static PCollection<GCSFileInfo> listGCSFiles(Pipeline p, VisionAnalyticsPipelineOptions options) {
    PCollection<GCSFileInfo> fileInfos;
    PCollection<Metadata> allFiles = p.begin()
        .apply("Get File List", Create.of(options.getFileList()))
        .apply("Match GCS Files", FileIO.matchAll());
    fileInfos = allFiles.apply(ParDo.of(new DoFn<Metadata, GCSFileInfo>() {
      private static final long serialVersionUID = 1L;

      @ProcessElement
      public void processElement(@Element Metadata metadata, OutputReceiver<GCSFileInfo> out) {
        out.output(GCSUtils.getFileInfo(metadata.resourceId().toString()));
      }
    }))
        .apply("Filter out non-image files",
            Filter.by((SerializableFunction<GCSFileInfo, Boolean>) fileName -> {
              totalFiles.inc();
              if (fileName.getUri().matches("(^.*\\.(" + ACCEPTED_FILE_PATTERN + ")$)")) {
                return true;
              }
              LOG.warn("File {} does not contain a valid extension", fileName);
              rejectedFiles.inc();
              return false;
            }));
    return fileInfos;
  }

  /**
   * Creates a map of well-known {@link MLApiResponseProcessor}s.
   *
   * If additional processors are needed they should be configured in this method.
   */
  private static Map<String, MLApiResponseProcessor> configureProcessors(
      VisionAnalyticsPipelineOptions options) {
    Map<String, MLApiResponseProcessor> result = new HashMap<>();

    // Image processors ------------------------------------------------------------------------------

    String tableName = options.getImageLabelAnnotationTable();
    result.put(tableName, new LabelAnnotationProcessor(tableName));

    tableName = options.getImageLandmarkAnnotationTable();
    result.put(tableName, new LandmarkAnnotationProcessor(tableName));

    tableName = options.getImageLogoAnnotationTable();
    result.put(tableName, new LogoAnnotationProcessor(tableName));

    tableName = options.getImageFaceAnnotationTable();
    result.put(tableName, new FaceAnnotationProcessor(tableName));

    tableName = options.getImagePropertiesTable();
    result.put(tableName, new ImagePropertiesProcessor(tableName));

    tableName = options.getImageCropHintAnnotationTable();
    result.put(tableName, new CropHintAnnotationProcessor(tableName));

    tableName = options.getErrorLogTable();
    result.put(tableName, new ErrorProcessor(tableName));

    // Video processors ------------------------------------------------------------------------------

    tableName = options.getVideoObjectTrackingAnnotationTable();
    result.put(tableName, new VideoObjectTrackingAnnotationProcessor(tableName));

    tableName = options.getVideoLabelAnnotationTable();
    result.put(tableName, new VideoLabelAnnotationProcessor(tableName, options.getMetadataKeys()));

    return result;
  }
}
