#include <iostream>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <string>
#include <array>
#include <vector>
#include "rknn_api.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <queue>
#include <cmath>
#include <map>
#include <sys/time.h>
#include "opencv2/opencv.hpp"
using namespace std;

struct ModelConfig{
  string model_path = "yolov5.rknn";
  float conf_thres = 0.4;
  int nc = 80;
  float iou_thre = 0.1;
  std::array<int, 2> model_shape {640, 640};
  std::vector<int> anchors {10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116, 90, 156, 198, 373, 326};
  std::vector<int> grids {80,80,40,40,20,20};
};

struct ModelInput{
  void* data = NULL;
  int height=0;
  int width=0;
  int channel=1;
};

class BBox {
 public:
  BBox() = default;
  ~BBox() = default;

  BBox(float x1, float y1, float x2, float y2) {
    xmin_ = x1;
    ymin_ = y1;
    xmax_ = x2;
    ymax_ = y2;
  }

 public:
  float xmin_;
  float ymin_;
  float xmax_;
  float ymax_;
};

class Anchor {
 public:
  Anchor() = default;
  ~Anchor() = default;
  bool operator<(const Anchor &t) const { return score_ < t.score_; }
  bool operator>(const Anchor &t) const { return score_ > t.score_; }

 public:
  float score_;
  int class_index;                   // cls score
  BBox finalbox_;        // final box res
};

struct ModelOutput{
  std::vector<Anchor> res;
};

float sigmoid(float input){
  return 1.0 / (1.0 + exp(-input));
}

void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes) {

	if (boxes.size() == 0)
		return;
	std::vector<size_t> idx(boxes.size());

	for (unsigned i = 0; i < idx.size(); i++)
	{
		idx[i] = i;
	}

	sort(boxes.begin(), boxes.end(), std::greater<Anchor>());

	while (idx.size() > 0)
	{
		int good_idx = idx[0];
		filterOutBoxes.push_back(boxes[good_idx]);

		std::vector<size_t> tmp = idx;
		idx.clear();
		for (unsigned i = 1; i < tmp.size(); i++)
		{
			int tmp_i = tmp[i];
			float inter_x1 = std::max(boxes[good_idx].finalbox_.xmin_, boxes[tmp_i].finalbox_.xmin_);
			float inter_y1 = std::max(boxes[good_idx].finalbox_.ymin_, boxes[tmp_i].finalbox_.ymin_);
			float inter_x2 = std::min(boxes[good_idx].finalbox_.xmax_, boxes[tmp_i].finalbox_.xmax_);
			float inter_y2 = std::min(boxes[good_idx].finalbox_.ymax_, boxes[tmp_i].finalbox_.ymax_);

			float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
			float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

			float inter_area = w * h;
			float area_1 = (boxes[good_idx].finalbox_.xmax_ - boxes[good_idx].finalbox_.xmin_ + 1) * (boxes[good_idx].finalbox_.ymax_ - boxes[good_idx].finalbox_.ymin_ + 1);
			float area_2 = (boxes[tmp_i].finalbox_.xmax_ - boxes[tmp_i].finalbox_.xmin_ + 1) * (boxes[tmp_i].finalbox_.ymax_ - boxes[tmp_i].finalbox_.ymin_ + 1);
			float o = inter_area / (area_1 + area_2 - inter_area);
			if (o <= threshold)
				idx.push_back(tmp_i);
		}
	}
}

class Yolov5Detector{
public:
  Yolov5Detector(ModelConfig config);
  int Detect(ModelOutput& output,
    ModelInput input);
  ~Yolov5Detector(){
    if(ctx > 0)         rknn_destroy(ctx);
    if(model)           free(model);
  }
private:
  int PreProcess(cv::Mat& output, cv::Mat input);
  int PostProcess(std::vector<Anchor>& filtered_outputs,
    std::vector<float*> network_outputs);
private:
  void *model = NULL;
  rknn_input inputs[1];
  rknn_output outputs[3];
  rknn_tensor_attr output0_attr;
  rknn_context ctx = 0;
  ModelConfig cfg;
  int pad_left = 0;
  int pad_top = 0;
  float scale = 0.;
  int nc = 1;
};


Yolov5Detector::Yolov5Detector(ModelConfig config) {
  cfg = config;
  nc = cfg.nc;
  FILE* fp = fopen(config.model_path.data(), "rb");
  if(fp == NULL) {
        printf("fopen %s fail!\n", config.model_path.data());
        return;
  }
  fseek(fp, 0, SEEK_END);
  int model_len = ftell(fp);
  model = malloc(model_len);
  fseek(fp, 0, SEEK_SET);
  if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", config.model_path.data());
        free(model);
        return;
  }
  int ret = 0;
  ret = rknn_init(&ctx, model, model_len, RKNN_FLAG_PRIOR_MEDIUM | RKNN_FLAG_COLLECT_PERF_MASK,NULL);
  if(ret < 0) {
      printf("rknn_init fail! ret=%d\n", ret);
      if(ctx > 0)         rknn_destroy(ctx);
      if(model)           free(model);
      if(fp)              fclose(fp);
      return;
  }
  output0_attr.index = 1;
  ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output0_attr, sizeof(output0_attr));
  if(ret < 0) {
      printf("rknn_query fail! ret=%d\n", ret);
      if(ctx > 0)         rknn_destroy(ctx);
      if(model)           free(model);
      if(fp)              fclose(fp);
      return;
  }
  outputs[0].want_float = true;
  outputs[0].is_prealloc = false;
  outputs[1].want_float = true;
  outputs[1].is_prealloc = false;
  outputs[2].want_float = true;
  outputs[2].is_prealloc = false;
  fclose(fp);

}

int Yolov5Detector::Detect(ModelOutput& output, ModelInput input){
  cv::Mat model_input;
  cv::Mat img_(input.height, input.width, CV_8UC3, input.data);
  PreProcess(model_input, img_);
  // cv::imwrite("tmp.jpg", model_input);
  double start = cv::getTickCount();
  int ret = 0;
  inputs[0].index = 0;
  inputs[0].buf = model_input.data;
  inputs[0].size = model_input.cols * model_input.rows * model_input.channels();
  inputs[0].pass_through = false;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  ret = rknn_inputs_set(ctx, 1, inputs);
  if(ret < 0) {
      printf("rknn_input_set fail! ret=%d\n", ret);
      if(ctx > 0)         rknn_destroy(ctx);
      if(model)           free(model);
      return ret;
  }
  ret = rknn_run(ctx, nullptr);
  if(ret < 0) {
      printf("rknn_run fail! ret=%d\n", ret);
      if(ctx > 0)         rknn_destroy(ctx);
      if(model)           free(model);
      return ret;
  }
  ret = rknn_outputs_get(ctx, 3, outputs, nullptr);
  if(ret < 0) {
      printf("rknn_outputs_get fail! ret=%d\n", ret);
      if(ctx > 0)         rknn_destroy(ctx);
      if(model)           free(model);
      return ret;
  }
  double end = cv::getTickCount();
  printf("rknn inference time cost: %f\n", (end - start) / cv::getTickFrequency() * 1000.);    
  std::vector<float*> network_outputs {
    (float*)outputs[0].buf,
    (float*)outputs[1].buf,
    (float*)outputs[2].buf
  };
  std::vector<Anchor> filtered_outputs;
  PostProcess(filtered_outputs, network_outputs);

  output.res.assign(filtered_outputs.begin(), filtered_outputs.end());
  std::cout << "first filter size: " << filtered_outputs.size() << std::endl;
  return 0;
}

int Yolov5Detector::PreProcess(cv::Mat& output, cv::Mat input){
  output = cv::Mat(input);
  int img_h = input.rows;
  int img_w = input.cols;

  float r = std::min(1.0 * cfg.model_shape[0] / img_w,
     1.0 * cfg.model_shape[1] / img_h);

  int new_unpad_w = int(img_w * r + 0.5);
  int new_unpad_h = int(img_h * r + 0.5);
  float dw = cfg.model_shape[0] - new_unpad_w;
  float dh = cfg.model_shape[1] - new_unpad_h;

  dw /= 2;
  dh /= 2;

  if(!(new_unpad_w == img_w && new_unpad_h == img_h )){
    cv::resize(output, output, cv::Size(new_unpad_w, new_unpad_h));
  }

  int top    = int(dh + 0.4);
  int bottom = int(dh + 0.6);
  int left   = int(dw + 0.4);
  int right  = int(dw + 0.6);

  cv::copyMakeBorder(output, output, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
  pad_left = left;
  pad_top = top;
  scale = r;
  return 0;
}

int Yolov5Detector::PostProcess(std::vector<Anchor>& filtered_outputs,std::vector<float*> network_outputs){
  filtered_outputs.clear();
  std::map<int, std::vector<Anchor>> map_anchors;
  // std::vector<Anchor> tmp_anchors;
  int grid_w, grid_h;
  for(int o=0;o<3;++o){
    grid_w = cfg.grids[o*2];
    grid_h = cfg.grids[o*2+1];
    for(int a=0; a<3; ++a){
      for(int y=0; y<grid_h;++y){
        for(int x=0; x < grid_w; ++x){

          int pos = a*grid_w*grid_h*(5+nc)+y*grid_w*(5+nc)+x*(5+nc);
          float pred_x = sigmoid(network_outputs[o][0  + pos]);
          float pred_y = sigmoid(network_outputs[o][1  + pos]);
          float pred_w = sigmoid(network_outputs[o][2  + pos]);
          float pred_h = sigmoid(network_outputs[o][3  + pos]);
          float pred_conf = sigmoid(network_outputs[o][4  + pos]);
          if(pred_conf < cfg.conf_thres) continue;

          float max_pred_conf = 0.;
          int max_pred_conf_idx = 0;

          for(int i=0;i<nc;i++){
            if(pred_conf < cfg.conf_thres) continue;
            float pred_cla = sigmoid(network_outputs[o][i+5 + pos]);
            float tmp_score = pred_conf * pred_cla;
            if(max_pred_conf < tmp_score){
              max_pred_conf_idx = i;
              max_pred_conf = tmp_score;
            }
          }

          if(max_pred_conf < cfg.conf_thres){ //cfg.conf_thre_list[max_pred_conf_idx]){
            continue;
          }else{
            Anchor anchor;
            anchor.score_ = max_pred_conf;
            anchor.class_index = max_pred_conf_idx;
            float c_x = 1.0 * (pred_x * 2 - 0.5 + x) * cfg.model_shape[0] / grid_w;
            float c_y = 1.0 * (pred_y * 2 - 0.5 + y) * cfg.model_shape[0] / grid_w;
            float c_w = 4.0 * pred_w * pred_w * cfg.anchors[o*3*2 + a*2];
            float c_h = 4.0 * pred_h * pred_h * cfg.anchors[o*3*2 + a*2 +1];
            anchor.finalbox_.xmin_ = (c_x - 0.5 * c_w - pad_left) / scale;
            anchor.finalbox_.ymin_ = (c_y - 0.5 * c_h - pad_top) / scale;
            anchor.finalbox_.xmax_ = (c_x + 0.5 * c_w - pad_left) / scale;
            anchor.finalbox_.ymax_ = (c_y + 0.5 * c_h - pad_top) / scale;
            if(map_anchors.find(max_pred_conf_idx) == map_anchors.end()){
              std::vector<Anchor> tmp_anchor{anchor};
              map_anchors[max_pred_conf_idx] = tmp_anchor;
            }else{
              map_anchors[max_pred_conf_idx].push_back(anchor);
            }
          }
        }
      }
    }
  }
  for(int i=0;i<cfg.nc;i++){
    if(map_anchors.find(i) != map_anchors.end()){
      //std::cout << map_anchors[i].size() << std::endl;
      nms_cpu(map_anchors[i], cfg.iou_thre, filtered_outputs);
    }
  }
  return 0;
}

int main(int argc, char *argv[]){
    ModelConfig config;
    config.model_path = "model/yolov5s-640-640.rknn";
    if(argc > 1){
      if(argv[1][0] == 'q'){
        config.model_path = "yolov5_q.rknn";
      }
    }
    Yolov5Detector detector(config);
    printf("model init ok \n");

    ModelOutput output;
    ModelInput input;


    //cv::VideoCapture cap("/data/test/zhongzi001.avi");
    cv::Mat img;

    img = cv::imread("model/test.jpg");
    //cap.read(img);

    input.data = img.data;
    input.width = img.cols;
    input.height = img.rows;
    input.channel = 3;
    double start = cv::getTickCount();
    detector.Detect(output, input);
    double end = cv::getTickCount();
    printf("time cost: %f\n", (end - start) / cv::getTickFrequency() * 1000.);
    for(int i=0;i<output.res.size();i++){
        BBox bbox = output.res[i].finalbox_;
        cv::Rect rect(bbox.xmin_,
            bbox.ymin_,
            bbox.xmax_ - bbox.xmin_,
            bbox.ymax_ - bbox.ymin_);
        cv::rectangle(img, rect, cv::Scalar(255),2);
    }
    cv::imwrite("model/out.jpg", img);
    
    // while(1){
    //   img = cv::imread("model/test.jpg");
    //   //cap.read(img);

    //   input.data = img.data;
    //   input.width = img.cols;
    //   input.height = img.rows;
    //   input.channel = 3;
    //   double start = cv::getTickCount();
    //   detector.Detect(output, input);
    //   double end = cv::getTickCount();
    //   printf("time cost: %f\n", (end - start) / cv::getTickFrequency() * 1000.);
    //   for(int i=0;i<output.res.size();i++){
    //     BBox bbox = output.res[i].finalbox_;
    //     cv::Rect rect(bbox.xmin_,
    //       bbox.ymin_,
    //       bbox.xmax_ - bbox.xmin_,
    //       bbox.ymax_ - bbox.ymin_);
    //     cv::rectangle(img, rect, cv::Scalar(255),2);
    //   }
    //   cv::imwrite("model/out.jpg", img);
    // }
    return 0;
}
