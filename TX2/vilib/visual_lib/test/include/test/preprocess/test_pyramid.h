/*
 * Tests for image pyramid functionalities
 * test_pyramid.h
 *
 * Copyright (c) 2019-2020 Balazs Nagy,
 * Robotics and Perception Group, University of Zurich
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "test/test_base.h"

class TestPyramid : public TestBase {
public:
  TestPyramid(const char * image_path);
  ~TestPyramid(void);
protected:
  bool run(void);
  void preallocate_pyramid_gpu(std::vector<unsigned char*> & pyramid_image,
                               std::vector<std::size_t> & pyramid_width,
                               std::vector<std::size_t> & pyramid_height,
                               std::vector<std::size_t> & pyramid_pitch,
                               std::size_t pyramid_levels);
  void preallocate_pyramid_cpu(std::vector<cv::Mat> & pyramid_image_cpu,
                               std::size_t pyramid_levels);
  void deallocate_pyramid_gpu(std::vector<unsigned char*> & pyramid_image);
  void copy_pyramid_from_gpu(std::vector<unsigned char*> & pyramid_image_gpu,
                             std::vector<cv::Mat> & pyramid_image_cpu,
                             std::vector<std::size_t> & pyramid_pitch_gpu,
                             std::size_t pyramid_levels);
};
