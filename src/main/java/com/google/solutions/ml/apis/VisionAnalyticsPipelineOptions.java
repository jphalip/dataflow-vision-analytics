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

import com.google.cloud.videointelligence.v1p3beta1.StreamingFeature;
import com.google.cloud.vision.v1.Feature;
import java.util.List;
import java.util.Set;

import org.apache.beam.runners.dataflow.options.DataflowPipelineOptions;
import org.apache.beam.sdk.options.Default;
import org.apache.beam.sdk.options.Description;
import org.apache.beam.sdk.options.Validation;

/**
 * Interface to store pipeline options provided by the user
 */
public interface VisionAnalyticsPipelineOptions extends DataflowPipelineOptions {

  @Description("Pub/Sub subscription ID to receive input Cloud Storage notifications from")
  String getInputNotificationSubscription();

  void setInputNotificationSubscription(String value);

  @Description("Google Cloud Storage files to process")
  List<String> getFileList();

  void setFileList(List<String> value);

  @Description("Key range")
  @Default.Integer(1)
  Integer getKeyRange();

  void setKeyRange(Integer value);

  @Description("Image annotation request batch size")
  @Default.Integer(1)
  Integer getBatchSize();

  void setBatchSize(Integer value);

  @Description("Window interval in seconds (default is 5)")
  @Default.Integer(5)
  Integer getWindowInterval();

  void setWindowInterval(Integer value);

  @Description("BigQuery dataset")
  @Validation.Required
  String getDatasetName();

  void setDatasetName(String value);

  @Description("Vision API features to use")
  List<Feature.Type> getImageFeatures();

  void setImageFeatures(List<Feature.Type> value);

  @Description("Streaming video feature")
  StreamingFeature getVideoFeature();

  void setVideoFeature(StreamingFeature value);

  @Description("Simulate annotations")
  @Default.Boolean(false)
  boolean isSimulate();

  void setSimulate(boolean value);

  @Description("Collect batch data")
  @Default.Boolean(false)
  boolean isCollectBatchData();

  void setCollectBatchData(boolean value);

  @Description("Table name for image label annotations")
  @Default.String("image_label_annotation")
  String getImageLabelAnnotationTable();

  void setImageLabelAnnotationTable(String value);

  @Description("Table name for image landmark annotations")
  @Default.String("image_landmark_annotation")
  String getImageLandmarkAnnotationTable();

  void setImageLandmarkAnnotationTable(String value);

  @Description("Table name for image logo annotations")
  @Default.String("image_logo_annotation")
  String getImageLogoAnnotationTable();

  void setImageLogoAnnotationTable(String value);

  @Description("Table name for image face annotations")
  @Default.String("image_face_annotation")
  String getImageFaceAnnotationTable();

  void setImageFaceAnnotationTable(String value);

  @Description("Table name for image properties")
  @Default.String("image_properties")
  String getImagePropertiesTable();

  void setImagePropertiesTable(String value);

  @Description("Table name for image crop hint annotations")
  @Default.String("image_crop_hint_annotation")
  String getImageCropHintAnnotationTable();

  void setImageCropHintAnnotationTable(String value);

  @Description("Table name for video object tracking annotations")
  @Default.String("video_object_tracking_annotation")
  String getVideoObjectTrackingAnnotationTable();

  void setVideoObjectTrackingAnnotationTable(String value);

  @Description("Table name for video label annotations")
  @Default.String("video_label_annotation")
  String getVideoLabelAnnotationTable();

  void setVideoLabelAnnotationTable(String value);

  @Description("Table name for error logs")
  @Default.String("error_log")
  String getErrorLogTable();

  void setErrorLogTable(String value);

  @Description("GCS metadata values to store in BigQuery")
  Set<String> getMetadataKeys();

  void setMetadataKeys(Set<String> value);
}
