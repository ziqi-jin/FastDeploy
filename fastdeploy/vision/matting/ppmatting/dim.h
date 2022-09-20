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
#pragma once
#include "fastdeploy/vision/matting/ppmatting/ppmatting.h"

namespace fastdeploy {
namespace vision {
namespace matting {

class FASTDEPLOY_DECL DIM : public PPMatting {
 public:
  DIM(const std::string& model_file, const std::string& params_file,
      const std::string& config_file,
      const RuntimeOption& custom_option = RuntimeOption(),
      const Frontend& model_format = Frontend::PADDLE);

  std::string ModelName() const { return "DIM"; }

  virtual bool Predict(cv::Mat* im, MattingResult* result);

 private:
  bool Initialize();

  bool BuildPreprocessPipelineFromConfig();

  bool Preprocess(Mat* mat, FDTensor* outputs,
                  std::map<std::string, std::array<int, 2>>* im_info);

  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;
};

}  // namespace matting
}  // namespace vision
}  // namespace fastdeploy
