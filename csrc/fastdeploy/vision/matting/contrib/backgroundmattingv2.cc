// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/matting/contrib/modnet.h"
#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {

namespace vision {

namespace matting {

void BackgroundMattingv2::StrideCheck(Mat* mat,
                                      const std::vector<float>& color) {
  int h = mat.rows;
  int w = mat.cols;

  if (h % stride != 0 || w % stride != 0) {
    int align_h = stride * ((h - 1) / stride + 1);
    int align_w = stride * ((w - 1) / stride + 1);
    int pad_h = align_h - h;  // >= 0
    int pad_w = align_w - w;  // >= 0
    if (pad_h > 0 || pad_w > 0) {
      float half_h = pad_h * 1.0 / 2;
      int top = int(round(half_h - 0.1));
      int bottom = int(round(half_h + 0.1));
      float half_w = pad_w * 1.0 / 2;
      int left = int(round(half_w - 0.1));
      int right = int(round(half_w + 0.1));
      Pad::Run(mat, top, bottom, left, right, color);
    }
    std::cout << "found size is not aligned, the size will be changed as (h="
              << mat.rows << ", w=" << mat.cols << std::endl;
  }
  // TODO delete
  std::cout << mat.rows << std::endl;
  std::cout << mat.cols << std::endl;
}

BackgroundMattingv2::BackgroundMattingv2(const std::string& model_file,
                                         const std::string& params_file,
                                         const RuntimeOption& custom_option,
                                         const Frontend& model_format) {
  if (model_format == Frontend::ONNX) {
    valid_cpu_backends = {Backend::ORT};  // 指定可用的CPU后端
    valid_gpu_backends = {Backend::ORT, Backend::TRT};  // 指定可用的GPU后端
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool BackgroundMattingv2::Initialize() {
  // parameters for preprocess
  size = {-1, -1};
  alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  beta = {0.0f, 0.0f, 0.0f};
  swap_rb = true;
  padding_value = {114.0, 114.0, 114.0};

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }

  is_dynamic_input_ = false;
  is_default_size_ = true;
  auto shape = InputInfoOfRuntime(0).shape;
  for (int i = 0; i < shape.size(); ++i) {
    // if height or width is dynamic
    if (i >= 2 && shape[i] <= 0) {
      is_dynamic_input_ = true;
      break;
    }
  }
  return true;
}

bool BackgroundMattingv2::Preprocess(
    Mat* mat, FDTensor* output,
    std::map<std::string, std::array<int, 2>>* im_info) {
  // 1. Resize
  // 2. BGR2RGB
  // 3. Normalize
  // 4. HWC2CHW
  // int resize_w = size[0];
  // int resize_h = size[1];
  // if (resize_h != mat->Height() || resize_w != mat->Width()) {
  //   Resize::Run(mat, resize_w, resize_h);
  // }
  if (is_default_size_) {
    if (is_dynamic_input_) {
    } else {
    }
  } else {
    if (is_dynamic_input_) {
    } else {
    }
  }
  // stride check
  BackgroundMattingv2::StrideCheck(mat, padding_value);
  // BGR2RGB
  if (swap_rb) {
    BGR2RGB::Run(mat);
  }

  Convert::Run(mat, alpha, beta);
  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {mat->Height(), mat->Width()};

  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");

  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);  // reshape to n, h, w, c
  return true;
}

bool BackgroundMattingv2::Postprocess(
    std::vector<FDTensor>& infer_result, MattingResult* result,
    const std::map<std::string, std::array<int, 2>>& im_info) {
  FDASSERT((infer_result.size() == 1),
           "The default number of output tensor must be 1 according to "
           "modnet.");
  FDTensor& alpha_tensor = infer_result.at(0);  // (1,h,w,1)
  FDASSERT((alpha_tensor.shape[0] == 1), "Only support batch =1 now.");
  if (alpha_tensor.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }

  // 先获取alpha并resize (使用opencv)
  auto iter_ipt = im_info.find("input_shape");
  auto iter_out = im_info.find("output_shape");
  FDASSERT(iter_out != im_info.end() && iter_ipt != im_info.end(),
           "Cannot find input_shape or output_shape from im_info.");
  int out_h = iter_out->second[0];
  int out_w = iter_out->second[1];
  int ipt_h = iter_ipt->second[0];
  int ipt_w = iter_ipt->second[1];

  // TODO: 需要修改成FDTensor或Mat的运算 现在依赖cv::Mat
  float* alpha_ptr = static_cast<float*>(alpha_tensor.Data());
  cv::Mat alpha_zero_copy_ref(out_h, out_w, CV_32FC1, alpha_ptr);
  Mat alpha_resized(alpha_zero_copy_ref);  // ref-only, zero copy.
  if ((out_h != ipt_h) || (out_w != ipt_w)) {
    // already allocated a new continuous memory after resize.
    // cv::resize(alpha_resized, alpha_resized, cv::Size(ipt_w, ipt_h));
    Resize::Run(&alpha_resized, ipt_w, ipt_h, -1, -1);
  }

  result->Clear();
  // note: must be setup shape before Resize
  result->contain_foreground = false;
  // 和输入原图大小对应的alpha
  result->shape = {static_cast<int64_t>(ipt_h), static_cast<int64_t>(ipt_w)};
  int numel = ipt_h * ipt_w;
  int nbytes = numel * sizeof(float);
  result->Resize(numel);
  std::memcpy(result->alpha.data(), alpha_resized.GetCpuMat()->data, nbytes);
  return true;
}

bool BackgroundMattingv2::Predict(cv::Mat* im, MattingResult* result) {
#ifdef FASTDEPLOY_DEBUG
  TIMERECORD_START(0)
#endif

  Mat mat(*im);
  std::vector<FDTensor> input_tensors(1);

  std::map<std::string, std::array<int, 2>> im_info;
  // Record the shape of image and the shape of preprocessed image
  im_info["input_shape"] = {mat.Height(), mat.Width()};
  im_info["output_shape"] = {mat.Height(), mat.Width()};
  // if the size is modified by user, is_default_size_ will be false
  if (size[0] != -1 || size[1] != -1) {
    is_default_size_ = false
  }

  if (!Preprocess(&mat, &input_tensors[0], &im_info)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }

#ifdef FASTDEPLOY_DEBUG
  TIMERECORD_END(0, "Preprocess")
  TIMERECORD_START(1)
#endif

  input_tensors[0].name = InputInfoOfRuntime(0).name;
  std::vector<FDTensor> output_tensors;
  if (!Infer(input_tensors, &output_tensors)) {
    FDERROR << "Failed to inference." << std::endl;
    return false;
  }
#ifdef FASTDEPLOY_DEBUG
  TIMERECORD_END(1, "Inference")
  TIMERECORD_START(2)
#endif

  if (!Postprocess(output_tensors, result, im_info)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }

#ifdef FASTDEPLOY_DEBUG
  TIMERECORD_END(2, "Postprocess")
#endif
  return true;
}

}  // namespace matting
}  // namespace vision
}  // namespace fastdeploy