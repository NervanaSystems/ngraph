//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
//*****************************************************************************

#include <array>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "benchmark.hpp"
#include "benchmark_utils.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

class TensorCollection
{
public:
    vector<shared_ptr<runtime::HostTensor>> parameter_data;
    vector<shared_ptr<runtime::HostTensor>> result_data;

    vector<shared_ptr<runtime::Tensor>> input_tensors;
    vector<shared_ptr<runtime::Tensor>> output_tensors;

private:
};

static mutex s_mutex;
static condition_variable s_condition;
static size_t current_iteration = 0;
static size_t s_iterations;
static size_t s_warmup_iterations;
static stopwatch s_timer;

static void
    thread_entry(runtime::Executable* exec, const TensorCollection& tensors, size_t pipeline_stage)
{
    const vector<shared_ptr<runtime::Tensor>>& args = tensors.input_tensors;
    const vector<shared_ptr<runtime::Tensor>>& results = tensors.output_tensors;
    while (current_iteration < s_iterations + s_warmup_iterations)
    {
        // std::cout << "current_iteration: " << current_iteration << " [" << pipeline_stage << ":"
        //           << std::this_thread::get_id() << "] before lock" << std::endl;
        unique_lock<mutex> lock(s_mutex);
        // std::cout << "current_iteration: " << current_iteration << " [" << pipeline_stage << ":"
        //           << std::this_thread::get_id() << "] past lock" << std::endl;
        if ((current_iteration & 1) != pipeline_stage)
        {
            // std::cout << "current_iteration: " << current_iteration << " [" << pipeline_stage << ":"
            //           << std::this_thread::get_id() << "] before wait" << std::endl;
            s_condition.wait(lock);
            // std::cout << "current_iteration: " << current_iteration << " [" << pipeline_stage << ":"
            //           << std::this_thread::get_id() << "] after wait" << std::endl;
        }
        else
        {
            if (current_iteration == s_warmup_iterations)
            {
                s_timer.start();
            }
            // Write just before we invoke call.
            // This will overlap based on early notification before call below. 
            for (size_t arg_index = 0; arg_index < args.size(); arg_index++)
            {
                const shared_ptr<runtime::Tensor>& arg = args[arg_index];
                if (arg->get_stale())
                {
                    const shared_ptr<runtime::HostTensor>& data =
                        tensors.parameter_data[arg_index];
                    // Different data each time
                    // random_init(data);
                    arg->write(data->get_data_ptr(),
                                data->get_element_count() * data->get_element_type().size());
                }
            }
            bool output_iteration = (current_iteration % 10 == 0);
            if (output_iteration)
            {
                std::cout << "Iteration " << current_iteration << std::endl;
            }
            bool checkpoint = (current_iteration % 100 == 0);

            // Kick off next stage of pipeline overlapping this one by:
            // * Bumping current_iteration
            // * Notifying (releases other thread from wait)
            // * Unlocking (allows other thread to start section above)
            current_iteration++;
            s_condition.notify_all();
            lock.unlock();

            // our turn to run
            exec->call(results, args);
            for (size_t result_index = 0; result_index < results.size(); result_index++)
            {
                const shared_ptr<runtime::HostTensor>& data = tensors.result_data[result_index];
                const shared_ptr<runtime::Tensor>& result = results[result_index];
                if (result->get_stale() || checkpoint)
                {
                    result->read(data->get_data_ptr(),
                                 data->get_element_count() * data->get_element_type().size());
                }
            }
            if (current_iteration == (s_iterations + s_warmup_iterations - 1))
            {
                s_timer.stop();
            }
        }
    }
}

struct InputOutputCorrespondence
{
    std::string m_op_name;
    size_t m_input_index;
    size_t m_output_index;
};

std::vector<InputOutputCorrespondence> s_input_output_correspondences = {
    {"bn_conv1_mean", 3, 0},
    {"bn_conv1_variance", 5, 1},
    {"bn2a_branch2a_mean", 8, 2},
    {"bn2a_branch2a_variance", 10, 3},
    {"bn2a_branch2b_mean", 13, 4},
    {"bn2a_branch2b_variance", 15, 5},
    {"bn2a_branch2c_mean", 18, 6},
    {"bn2a_branch2c_variance", 20, 7},
    {"bn2a_branch1_mean", 23, 8},
    {"bn2a_branch1_variance", 25, 9},
    {"bn2b_branch2a_mean", 28, 10},
    {"bn2b_branch2a_variance", 30, 11},
    {"bn2b_branch2b_mean", 33, 12},
    {"bn2b_branch2b_variance", 35, 13},
    {"bn2b_branch2c_mean", 38, 14},
    {"bn2b_branch2c_variance", 40, 15},
    {"bn2c_branch2a_mean", 43, 16},
    {"bn2c_branch2a_variance", 45, 17},
    {"bn2c_branch2b_mean", 48, 18},
    {"bn2c_branch2b_variance", 50, 19},
    {"bn2c_branch2c_mean", 53, 20},
    {"bn2c_branch2c_variance", 55, 21},
    {"bn3a_branch2a_mean", 58, 22},
    {"bn3a_branch2a_variance", 60, 23},
    {"bn3a_branch2b_mean", 63, 24},
    {"bn3a_branch2b_variance", 65, 25},
    {"bn3a_branch2c_mean", 68, 26},
    {"bn3a_branch2c_variance", 70, 27},
    {"bn3a_branch1_mean", 73, 28},
    {"bn3a_branch1_variance", 75, 29},
    {"bn3b_branch2a_mean", 78, 30},
    {"bn3b_branch2a_variance", 80, 31},
    {"bn3b_branch2b_mean", 83, 32},
    {"bn3b_branch2b_variance", 85, 33},
    {"bn3b_branch2c_mean", 88, 34},
    {"bn3b_branch2c_variance", 90, 35},
    {"bn3c_branch2a_mean", 93, 36},
    {"bn3c_branch2a_variance", 95, 37},
    {"bn3c_branch2b_mean", 98, 38},
    {"bn3c_branch2b_variance", 100, 39},
    {"bn3c_branch2c_mean", 103, 40},
    {"bn3c_branch2c_variance", 105, 41},
    {"bn3d_branch2a_mean", 108, 42},
    {"bn3d_branch2a_variance", 110, 43},
    {"bn3d_branch2b_mean", 113, 44},
    {"bn3d_branch2b_variance", 115, 45},
    {"bn3d_branch2c_mean", 118, 46},
    {"bn3d_branch2c_variance", 120, 47},
    {"bn4a_branch2a_mean", 123, 48},
    {"bn4a_branch2a_variance", 125, 49},
    {"bn4a_branch2b_mean", 128, 50},
    {"bn4a_branch2b_variance", 130, 51},
    {"bn4a_branch2c_mean", 133, 52},
    {"bn4a_branch2c_variance", 135, 53},
    {"bn4a_branch1_mean", 138, 54},
    {"bn4a_branch1_variance", 140, 55},
    {"bn4b_branch2a_mean", 143, 56},
    {"bn4b_branch2a_variance", 145, 57},
    {"bn4b_branch2b_mean", 148, 58},
    {"bn4b_branch2b_variance", 150, 59},
    {"bn4b_branch2c_mean", 153, 60},
    {"bn4b_branch2c_variance", 155, 61},
    {"bn4c_branch2a_mean", 158, 62},
    {"bn4c_branch2a_variance", 160, 63},
    {"bn4c_branch2b_mean", 163, 64},
    {"bn4c_branch2b_variance", 165, 65},
    {"bn4c_branch2c_mean", 168, 66},
    {"bn4c_branch2c_variance", 170, 67},
    {"bn4d_branch2a_mean", 173, 68},
    {"bn4d_branch2a_variance", 175, 69},
    {"bn4d_branch2b_mean", 178, 70},
    {"bn4d_branch2b_variance", 180, 71},
    {"bn4d_branch2c_mean", 183, 72},
    {"bn4d_branch2c_variance", 185, 73},
    {"bn4e_branch2a_mean", 188, 74},
    {"bn4e_branch2a_variance", 190, 75},
    {"bn4e_branch2b_mean", 193, 76},
    {"bn4e_branch2b_variance", 195, 77},
    {"bn4e_branch2c_mean", 198, 78},
    {"bn4e_branch2c_variance", 200, 79},
    {"bn4f_branch2a_mean", 203, 80},
    {"bn4f_branch2a_variance", 205, 81},
    {"bn4f_branch2b_mean", 208, 82},
    {"bn4f_branch2b_variance", 210, 83},
    {"bn4f_branch2c_mean", 213, 84},
    {"bn4f_branch2c_variance", 215, 85},
    {"bn5a_branch2a_mean", 218, 86},
    {"bn5a_branch2a_variance", 220, 87},
    {"bn5a_branch2b_mean", 223, 88},
    {"bn5a_branch2b_variance", 225, 89},
    {"bn5a_branch2c_mean", 228, 90},
    {"bn5a_branch2c_variance", 230, 91},
    {"bn5a_branch1_mean", 233, 92},
    {"bn5a_branch1_variance", 235, 93},
    {"bn5b_branch2a_mean", 238, 94},
    {"bn5b_branch2a_variance", 240, 95},
    {"bn5b_branch2b_mean", 243, 96},
    {"bn5b_branch2b_variance", 245, 97},
    {"bn5b_branch2c_mean", 248, 98},
    {"bn5b_branch2c_variance", 250, 99},
    {"bn5c_branch2a_mean", 253, 100},
    {"bn5c_branch2a_variance", 255, 101},
    {"bn5c_branch2b_mean", 258, 102},
    {"bn5c_branch2b_variance", 260, 103},
    {"bn5c_branch2c_mean", 263, 104},
    {"bn5c_branch2c_variance", 265, 105},
    {"bn2a_branch1_offset", 22, 110},
    {"bn2a_branch1_offset_velocity_0", 270, 111},
    {"bn2a_branch1_scale", 24, 112},
    {"bn2a_branch1_scale_velocity_0", 271, 113},
    {"bn2a_branch2a_offset", 7, 114},
    {"bn2a_branch2a_offset_velocity_0", 272, 115},
    {"bn2a_branch2a_scale", 9, 116},
    {"bn2a_branch2a_scale_velocity_0", 273, 117},
    {"bn2a_branch2b_offset", 12, 118},
    {"bn2a_branch2b_offset_velocity_0", 274, 119},
    {"bn2a_branch2b_scale", 14, 120},
    {"bn2a_branch2b_scale_velocity_0", 275, 121},
    {"bn2a_branch2c_offset", 17, 122},
    {"bn2a_branch2c_offset_velocity_0", 276, 123},
    {"bn2a_branch2c_scale", 19, 124},
    {"bn2a_branch2c_scale_velocity_0", 277, 125},
    {"bn2b_branch2a_offset", 27, 126},
    {"bn2b_branch2a_offset_velocity_0", 278, 127},
    {"bn2b_branch2a_scale", 29, 128},
    {"bn2b_branch2a_scale_velocity_0", 279, 129},
    {"bn2b_branch2b_offset", 32, 130},
    {"bn2b_branch2b_offset_velocity_0", 280, 131},
    {"bn2b_branch2b_scale", 34, 132},
    {"bn2b_branch2b_scale_velocity_0", 281, 133},
    {"bn2b_branch2c_offset", 37, 134},
    {"bn2b_branch2c_offset_velocity_0", 282, 135},
    {"bn2b_branch2c_scale", 39, 136},
    {"bn2b_branch2c_scale_velocity_0", 283, 137},
    {"bn2c_branch2a_offset", 42, 138},
    {"bn2c_branch2a_offset_velocity_0", 284, 139},
    {"bn2c_branch2a_scale", 44, 140},
    {"bn2c_branch2a_scale_velocity_0", 285, 141},
    {"bn2c_branch2b_offset", 47, 142},
    {"bn2c_branch2b_offset_velocity_0", 286, 143},
    {"bn2c_branch2b_scale", 49, 144},
    {"bn2c_branch2b_scale_velocity_0", 287, 145},
    {"bn2c_branch2c_offset", 52, 146},
    {"bn2c_branch2c_offset_velocity_0", 288, 147},
    {"bn2c_branch2c_scale", 54, 148},
    {"bn2c_branch2c_scale_velocity_0", 289, 149},
    {"bn3a_branch1_offset", 72, 150},
    {"bn3a_branch1_offset_velocity_0", 290, 151},
    {"bn3a_branch1_scale", 74, 152},
    {"bn3a_branch1_scale_velocity_0", 291, 153},
    {"bn3a_branch2a_offset", 57, 154},
    {"bn3a_branch2a_offset_velocity_0", 292, 155},
    {"bn3a_branch2a_scale", 59, 156},
    {"bn3a_branch2a_scale_velocity_0", 293, 157},
    {"bn3a_branch2b_offset", 62, 158},
    {"bn3a_branch2b_offset_velocity_0", 294, 159},
    {"bn3a_branch2b_scale", 64, 160},
    {"bn3a_branch2b_scale_velocity_0", 295, 161},
    {"bn3a_branch2c_offset", 67, 162},
    {"bn3a_branch2c_offset_velocity_0", 296, 163},
    {"bn3a_branch2c_scale", 69, 164},
    {"bn3a_branch2c_scale_velocity_0", 297, 165},
    {"bn3b_branch2a_offset", 77, 166},
    {"bn3b_branch2a_offset_velocity_0", 298, 167},
    {"bn3b_branch2a_scale", 79, 168},
    {"bn3b_branch2a_scale_velocity_0", 299, 169},
    {"bn3b_branch2b_offset", 82, 170},
    {"bn3b_branch2b_offset_velocity_0", 300, 171},
    {"bn3b_branch2b_scale", 84, 172},
    {"bn3b_branch2b_scale_velocity_0", 301, 173},
    {"bn3b_branch2c_offset", 87, 174},
    {"bn3b_branch2c_offset_velocity_0", 302, 175},
    {"bn3b_branch2c_scale", 89, 176},
    {"bn3b_branch2c_scale_velocity_0", 303, 177},
    {"bn3c_branch2a_offset", 92, 178},
    {"bn3c_branch2a_offset_velocity_0", 304, 179},
    {"bn3c_branch2a_scale", 94, 180},
    {"bn3c_branch2a_scale_velocity_0", 305, 181},
    {"bn3c_branch2b_offset", 97, 182},
    {"bn3c_branch2b_offset_velocity_0", 306, 183},
    {"bn3c_branch2b_scale", 99, 184},
    {"bn3c_branch2b_scale_velocity_0", 307, 185},
    {"bn3c_branch2c_offset", 102, 186},
    {"bn3c_branch2c_offset_velocity_0", 308, 187},
    {"bn3c_branch2c_scale", 104, 188},
    {"bn3c_branch2c_scale_velocity_0", 309, 189},
    {"bn3d_branch2a_offset", 107, 190},
    {"bn3d_branch2a_offset_velocity_0", 310, 191},
    {"bn3d_branch2a_scale", 109, 192},
    {"bn3d_branch2a_scale_velocity_0", 311, 193},
    {"bn3d_branch2b_offset", 112, 194},
    {"bn3d_branch2b_offset_velocity_0", 312, 195},
    {"bn3d_branch2b_scale", 114, 196},
    {"bn3d_branch2b_scale_velocity_0", 313, 197},
    {"bn3d_branch2c_offset", 117, 198},
    {"bn3d_branch2c_offset_velocity_0", 314, 199},
    {"bn3d_branch2c_scale", 119, 200},
    {"bn3d_branch2c_scale_velocity_0", 315, 201},
    {"bn4a_branch1_offset", 137, 202},
    {"bn4a_branch1_offset_velocity_0", 316, 203},
    {"bn4a_branch1_scale", 139, 204},
    {"bn4a_branch1_scale_velocity_0", 317, 205},
    {"bn4a_branch2a_offset", 122, 206},
    {"bn4a_branch2a_offset_velocity_0", 318, 207},
    {"bn4a_branch2a_scale", 124, 208},
    {"bn4a_branch2a_scale_velocity_0", 319, 209},
    {"bn4a_branch2b_offset", 127, 210},
    {"bn4a_branch2b_offset_velocity_0", 320, 211},
    {"bn4a_branch2b_scale", 129, 212},
    {"bn4a_branch2b_scale_velocity_0", 321, 213},
    {"bn4a_branch2c_offset", 132, 214},
    {"bn4a_branch2c_offset_velocity_0", 322, 215},
    {"bn4a_branch2c_scale", 134, 216},
    {"bn4a_branch2c_scale_velocity_0", 323, 217},
    {"bn4b_branch2a_offset", 142, 218},
    {"bn4b_branch2a_offset_velocity_0", 324, 219},
    {"bn4b_branch2a_scale", 144, 220},
    {"bn4b_branch2a_scale_velocity_0", 325, 221},
    {"bn4b_branch2b_offset", 147, 222},
    {"bn4b_branch2b_offset_velocity_0", 326, 223},
    {"bn4b_branch2b_scale", 149, 224},
    {"bn4b_branch2b_scale_velocity_0", 327, 225},
    {"bn4b_branch2c_offset", 152, 226},
    {"bn4b_branch2c_offset_velocity_0", 328, 227},
    {"bn4b_branch2c_scale", 154, 228},
    {"bn4b_branch2c_scale_velocity_0", 329, 229},
    {"bn4c_branch2a_offset", 157, 230},
    {"bn4c_branch2a_offset_velocity_0", 330, 231},
    {"bn4c_branch2a_scale", 159, 232},
    {"bn4c_branch2a_scale_velocity_0", 331, 233},
    {"bn4c_branch2b_offset", 162, 234},
    {"bn4c_branch2b_offset_velocity_0", 332, 235},
    {"bn4c_branch2b_scale", 164, 236},
    {"bn4c_branch2b_scale_velocity_0", 333, 237},
    {"bn4c_branch2c_offset", 167, 238},
    {"bn4c_branch2c_offset_velocity_0", 334, 239},
    {"bn4c_branch2c_scale", 169, 240},
    {"bn4c_branch2c_scale_velocity_0", 335, 241},
    {"bn4d_branch2a_offset", 172, 242},
    {"bn4d_branch2a_offset_velocity_0", 336, 243},
    {"bn4d_branch2a_scale", 174, 244},
    {"bn4d_branch2a_scale_velocity_0", 337, 245},
    {"bn4d_branch2b_offset", 177, 246},
    {"bn4d_branch2b_offset_velocity_0", 338, 247},
    {"bn4d_branch2b_scale", 179, 248},
    {"bn4d_branch2b_scale_velocity_0", 339, 249},
    {"bn4d_branch2c_offset", 182, 250},
    {"bn4d_branch2c_offset_velocity_0", 340, 251},
    {"bn4d_branch2c_scale", 184, 252},
    {"bn4d_branch2c_scale_velocity_0", 341, 253},
    {"bn4e_branch2a_offset", 187, 254},
    {"bn4e_branch2a_offset_velocity_0", 342, 255},
    {"bn4e_branch2a_scale", 189, 256},
    {"bn4e_branch2a_scale_velocity_0", 343, 257},
    {"bn4e_branch2b_offset", 192, 258},
    {"bn4e_branch2b_offset_velocity_0", 344, 259},
    {"bn4e_branch2b_scale", 194, 260},
    {"bn4e_branch2b_scale_velocity_0", 345, 261},
    {"bn4e_branch2c_offset", 197, 262},
    {"bn4e_branch2c_offset_velocity_0", 346, 263},
    {"bn4e_branch2c_scale", 199, 264},
    {"bn4e_branch2c_scale_velocity_0", 347, 265},
    {"bn4f_branch2a_offset", 202, 266},
    {"bn4f_branch2a_offset_velocity_0", 348, 267},
    {"bn4f_branch2a_scale", 204, 268},
    {"bn4f_branch2a_scale_velocity_0", 349, 269},
    {"bn4f_branch2b_offset", 207, 270},
    {"bn4f_branch2b_offset_velocity_0", 350, 271},
    {"bn4f_branch2b_scale", 209, 272},
    {"bn4f_branch2b_scale_velocity_0", 351, 273},
    {"bn4f_branch2c_offset", 212, 274},
    {"bn4f_branch2c_offset_velocity_0", 352, 275},
    {"bn4f_branch2c_scale", 214, 276},
    {"bn4f_branch2c_scale_velocity_0", 353, 277},
    {"bn5a_branch1_offset", 232, 278},
    {"bn5a_branch1_offset_velocity_0", 354, 279},
    {"bn5a_branch1_scale", 234, 280},
    {"bn5a_branch1_scale_velocity_0", 355, 281},
    {"bn5a_branch2a_offset", 217, 282},
    {"bn5a_branch2a_offset_velocity_0", 356, 283},
    {"bn5a_branch2a_scale", 219, 284},
    {"bn5a_branch2a_scale_velocity_0", 357, 285},
    {"bn5a_branch2b_offset", 222, 286},
    {"bn5a_branch2b_offset_velocity_0", 358, 287},
    {"bn5a_branch2b_scale", 224, 288},
    {"bn5a_branch2b_scale_velocity_0", 359, 289},
    {"bn5a_branch2c_offset", 227, 290},
    {"bn5a_branch2c_offset_velocity_0", 360, 291},
    {"bn5a_branch2c_scale", 229, 292},
    {"bn5a_branch2c_scale_velocity_0", 361, 293},
    {"bn5b_branch2a_offset", 237, 294},
    {"bn5b_branch2a_offset_velocity_0", 362, 295},
    {"bn5b_branch2a_scale", 239, 296},
    {"bn5b_branch2a_scale_velocity_0", 363, 297},
    {"bn5b_branch2b_offset", 242, 298},
    {"bn5b_branch2b_offset_velocity_0", 364, 299},
    {"bn5b_branch2b_scale", 244, 300},
    {"bn5b_branch2b_scale_velocity_0", 365, 301},
    {"bn5b_branch2c_offset", 247, 302},
    {"bn5b_branch2c_offset_velocity_0", 366, 303},
    {"bn5b_branch2c_scale", 249, 304},
    {"bn5b_branch2c_scale_velocity_0", 367, 305},
    {"bn5c_branch2a_offset", 252, 306},
    {"bn5c_branch2a_offset_velocity_0", 368, 307},
    {"bn5c_branch2a_scale", 254, 308},
    {"bn5c_branch2a_scale_velocity_0", 369, 309},
    {"bn5c_branch2b_offset", 257, 310},
    {"bn5c_branch2b_offset_velocity_0", 370, 311},
    {"bn5c_branch2b_scale", 259, 312},
    {"bn5c_branch2b_scale_velocity_0", 371, 313},
    {"bn5c_branch2c_offset", 262, 314},
    {"bn5c_branch2c_offset_velocity_0", 372, 315},
    {"bn5c_branch2c_scale", 264, 316},
    {"bn5c_branch2c_scale_velocity_0", 373, 317},
    {"bn_conv1_offset", 2, 318},
    {"bn_conv1_offset_velocity_0", 374, 319},
    {"bn_conv1_scale", 4, 320},
    {"bn_conv1_scale_velocity_0", 375, 321},
    {"conv1_weights", 0, 322},
    {"conv1_weights_velocity_0", 376, 323},
    {"fc_0.b_0", 267, 324},
    {"fc_0.b_0_velocity_0", 377, 325},
    {"fc_0.w_0", 266, 326},
    {"fc_0.w_0_velocity_0", 378, 327},
    {"res2a_branch1_weights", 21, 328},
    {"res2a_branch1_weights_velocity_0", 379, 329},
    {"res2a_branch2a_weights", 6, 330},
    {"res2a_branch2a_weights_velocity_0", 380, 331},
    {"res2a_branch2b_weights", 11, 332},
    {"res2a_branch2b_weights_velocity_0", 381, 333},
    {"res2a_branch2c_weights", 16, 334},
    {"res2a_branch2c_weights_velocity_0", 382, 335},
    {"res2b_branch2a_weights", 26, 336},
    {"res2b_branch2a_weights_velocity_0", 383, 337},
    {"res2b_branch2b_weights", 31, 338},
    {"res2b_branch2b_weights_velocity_0", 384, 339},
    {"res2b_branch2c_weights", 36, 340},
    {"res2b_branch2c_weights_velocity_0", 385, 341},
    {"res2c_branch2a_weights", 41, 342},
    {"res2c_branch2a_weights_velocity_0", 386, 343},
    {"res2c_branch2b_weights", 46, 344},
    {"res2c_branch2b_weights_velocity_0", 387, 345},
    {"res2c_branch2c_weights", 51, 346},
    {"res2c_branch2c_weights_velocity_0", 388, 347},
    {"res3a_branch1_weights", 71, 348},
    {"res3a_branch1_weights_velocity_0", 389, 349},
    {"res3a_branch2a_weights", 56, 350},
    {"res3a_branch2a_weights_velocity_0", 390, 351},
    {"res3a_branch2b_weights", 61, 352},
    {"res3a_branch2b_weights_velocity_0", 391, 353},
    {"res3a_branch2c_weights", 66, 354},
    {"res3a_branch2c_weights_velocity_0", 392, 355},
    {"res3b_branch2a_weights", 76, 356},
    {"res3b_branch2a_weights_velocity_0", 393, 357},
    {"res3b_branch2b_weights", 81, 358},
    {"res3b_branch2b_weights_velocity_0", 394, 359},
    {"res3b_branch2c_weights", 86, 360},
    {"res3b_branch2c_weights_velocity_0", 395, 361},
    {"res3c_branch2a_weights", 91, 362},
    {"res3c_branch2a_weights_velocity_0", 396, 363},
    {"res3c_branch2b_weights", 96, 364},
    {"res3c_branch2b_weights_velocity_0", 397, 365},
    {"res3c_branch2c_weights", 101, 366},
    {"res3c_branch2c_weights_velocity_0", 398, 367},
    {"res3d_branch2a_weights", 106, 368},
    {"res3d_branch2a_weights_velocity_0", 399, 369},
    {"res3d_branch2b_weights", 111, 370},
    {"res3d_branch2b_weights_velocity_0", 400, 371},
    {"res3d_branch2c_weights", 116, 372},
    {"res3d_branch2c_weights_velocity_0", 401, 373},
    {"res4a_branch1_weights", 136, 374},
    {"res4a_branch1_weights_velocity_0", 402, 375},
    {"res4a_branch2a_weights", 121, 376},
    {"res4a_branch2a_weights_velocity_0", 403, 377},
    {"res4a_branch2b_weights", 126, 378},
    {"res4a_branch2b_weights_velocity_0", 404, 379},
    {"res4a_branch2c_weights", 131, 380},
    {"res4a_branch2c_weights_velocity_0", 405, 381},
    {"res4b_branch2a_weights", 141, 382},
    {"res4b_branch2a_weights_velocity_0", 406, 383},
    {"res4b_branch2b_weights", 146, 384},
    {"res4b_branch2b_weights_velocity_0", 407, 385},
    {"res4b_branch2c_weights", 151, 386},
    {"res4b_branch2c_weights_velocity_0", 408, 387},
    {"res4c_branch2a_weights", 156, 388},
    {"res4c_branch2a_weights_velocity_0", 409, 389},
    {"res4c_branch2b_weights", 161, 390},
    {"res4c_branch2b_weights_velocity_0", 410, 391},
    {"res4c_branch2c_weights", 166, 392},
    {"res4c_branch2c_weights_velocity_0", 411, 393},
    {"res4d_branch2a_weights", 171, 394},
    {"res4d_branch2a_weights_velocity_0", 412, 395},
    {"res4d_branch2b_weights", 176, 396},
    {"res4d_branch2b_weights_velocity_0", 413, 397},
    {"res4d_branch2c_weights", 181, 398},
    {"res4d_branch2c_weights_velocity_0", 414, 399},
    {"res4e_branch2a_weights", 186, 400},
    {"res4e_branch2a_weights_velocity_0", 415, 401},
    {"res4e_branch2b_weights", 191, 402},
    {"res4e_branch2b_weights_velocity_0", 416, 403},
    {"res4e_branch2c_weights", 196, 404},
    {"res4e_branch2c_weights_velocity_0", 417, 405},
    {"res4f_branch2a_weights", 201, 406},
    {"res4f_branch2a_weights_velocity_0", 418, 407},
    {"res4f_branch2b_weights", 206, 408},
    {"res4f_branch2b_weights_velocity_0", 419, 409},
    {"res4f_branch2c_weights", 211, 410},
    {"res4f_branch2c_weights_velocity_0", 420, 411},
    {"res5a_branch1_weights", 231, 412},
    {"res5a_branch1_weights_velocity_0", 421, 413},
    {"res5a_branch2a_weights", 216, 414},
    {"res5a_branch2a_weights_velocity_0", 422, 415},
    {"res5a_branch2b_weights", 221, 416},
    {"res5a_branch2b_weights_velocity_0", 423, 417},
    {"res5a_branch2c_weights", 226, 418},
    {"res5a_branch2c_weights_velocity_0", 424, 419},
    {"res5b_branch2a_weights", 236, 420},
    {"res5b_branch2a_weights_velocity_0", 425, 421},
    {"res5b_branch2b_weights", 241, 422},
    {"res5b_branch2b_weights_velocity_0", 426, 423},
    {"res5b_branch2c_weights", 246, 424},
    {"res5b_branch2c_weights_velocity_0", 427, 425},
    {"res5c_branch2a_weights", 251, 426},
    {"res5c_branch2a_weights_velocity_0", 428, 427},
    {"res5c_branch2b_weights", 256, 428},
    {"res5c_branch2b_weights_velocity_0", 429, 429},
    {"res5c_branch2c_weights", 261, 430},
    {"res5c_branch2c_weights_velocity_0", 430, 431}};

vector<runtime::PerformanceCounter> run_benchmark_pipelined(shared_ptr<Function> f,
                                                            const string& backend_name,
                                                            size_t iterations,
                                                            bool timing_detail,
                                                            int warmup_iterations,
                                                            bool copy_data)
{
    map<size_t, InputOutputCorrespondence> input_to_output_map;
    map<size_t, InputOutputCorrespondence> output_to_input_map;
    for (InputOutputCorrespondence& input_output_correspondence : s_input_output_correspondences)
    {
        size_t input_index = input_output_correspondence.m_input_index;
        size_t output_index = input_output_correspondence.m_output_index;
        auto& param = f->get_parameters()[input_index];
        auto& result = f->get_results()[output_index];

        // Sanity check correspondence
        if ((param->get_element_type() != result->get_element_type()) ||
            (param->get_shape() != result->get_shape()))
        {
            std::cout << "Correspondence verification failed for: "
                      << input_output_correspondence.m_op_name << " Parameter: " << input_index
                      << " not equivalent to Result: " << output_index << std::endl;
        }
        else
        {
            input_to_output_map[input_output_correspondence.m_input_index] =
                input_output_correspondence;
            output_to_input_map[input_output_correspondence.m_output_index] =
                input_output_correspondence;
        }
    }

    constexpr size_t pipeline_depth = 2;
    s_iterations = iterations;
    s_warmup_iterations = warmup_iterations;
    array<TensorCollection, pipeline_depth> tensor_collections;
    stopwatch timer;
    timer.start();
    auto backend = runtime::Backend::create(backend_name);
    auto exec = backend->compile(f, timing_detail);
    timer.stop();
    cout.imbue(locale(""));
    cout << "compile time: " << timer.get_milliseconds() << "ms" << endl;
    set_denormals_flush_to_zero();

    // Create random input data for all input tensors
    size_t param_index = 0;
    std::cout << "Creating input data..." << std::endl;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        auto correspondence_iterator = input_to_output_map.find(param_index);
        bool should_pipeline = (correspondence_iterator == output_to_input_map.end());
        if (should_pipeline)
        {
            // One input per pipeline
            for (size_t i = 0; i < pipeline_depth; i++)
            {
                auto tensor_data =
                    make_shared<runtime::HostTensor>(param->get_element_type(), param->get_shape());
                random_init(tensor_data);
                tensor_collections[i].parameter_data.push_back(tensor_data);
            }
        }
        else
        {
            // Single input
            auto tensor_data =
                make_shared<runtime::HostTensor>(param->get_element_type(), param->get_shape());
            random_init(tensor_data);
            for (size_t i = 0; i < pipeline_depth; i++)
            {
                tensor_collections[i].parameter_data.push_back(tensor_data);
            }
        }
        ++param_index;
    }

    std::cout << "Creating ouput data..." << std::endl;
    // Create output tensors for all outputs
    size_t result_index = 0;
    for (shared_ptr<Node> result : f->get_results())
    {
        auto correspondence_iterator = output_to_input_map.find(result_index);
        if (correspondence_iterator != output_to_input_map.end())
        {
            InputOutputCorrespondence& correspondence = correspondence_iterator->second;
            size_t input_index = correspondence.m_input_index;
            auto& param = f->get_parameters()[input_index];
            if ((param->get_element_type() != result->get_element_type()) ||
                (param->get_shape() != result->get_shape()))
            {
                std::cout << "Correspondence verification failed for: " << correspondence.m_op_name
                          << " Parameter: " << input_index
                          << " not equivalent to Result: " << result_index << std::endl;
                exit(0);
            }
            for (size_t i = 0; i < pipeline_depth; i++)
            {
                auto& input_tensor_data = tensor_collections[i].parameter_data[input_index];
                tensor_collections[i].result_data.push_back(input_tensor_data);
            }
        }
        else
        {
            for (size_t i = 0; i < pipeline_depth; i++)
            {
                auto tensor_data = make_shared<runtime::HostTensor>(result->get_element_type(),
                                                                    result->get_shape());
                tensor_collections[i].result_data.push_back(tensor_data);
            }
        }
        ++result_index;
    }

    std::cout << "Creating input tensors..." << std::endl;
    // Create input tensors for all Parameters
    array<vector<shared_ptr<runtime::Tensor>>, pipeline_depth> input_tensors_array;
    size_t input_index = 0;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        auto correspondence_iterator = input_to_output_map.find(input_index);
        bool should_pipeline = (correspondence_iterator == input_to_output_map.end());
        if (should_pipeline)
        {
            auto input_tensors = exec->create_input_tensor(input_index, pipeline_depth);
            for (size_t i = 0; i < pipeline_depth; i++)
            {
                input_tensors[i]->set_stale(true);
                tensor_collections[i].input_tensors.push_back(input_tensors[i]);
            }
        }
        else
        {
            auto input_tensors = exec->create_input_tensor(input_index, 1);
            // Abuse staleness to indicate whether we're pipelining
            input_tensors[0]->set_stale(false);
            for (size_t i = 0; i < pipeline_depth; i++)
            {
                tensor_collections[i].input_tensors.push_back(input_tensors[0]);
            }
        }
        ++input_index;
    }

    std::cout << "Creating output tensors..." << std::endl;
    // Create output tensors for all Results
    array<vector<shared_ptr<runtime::Tensor>>, pipeline_depth> output_tensors_array;
    size_t output_index = 0;
    for (shared_ptr<Node> result : f->get_results())
    {
        auto correspondence_iterator = output_to_input_map.find(output_index);
        bool should_pipeline = (correspondence_iterator == output_to_input_map.end());
        if (should_pipeline)
        {
            auto output_tensors = exec->create_output_tensor(output_index, pipeline_depth);
            for (size_t i = 0; i < pipeline_depth; i++)
            {
                output_tensors[i]->set_stale(true);
                tensor_collections[i].output_tensors.push_back(output_tensors[i]);
            }
        }
        else
        {
            InputOutputCorrespondence& correspondence = correspondence_iterator->second;
            input_index = correspondence.m_input_index;
            for (size_t i = 0; i < pipeline_depth; i++)
            {
                tensor_collections[i].output_tensors.push_back(
                    tensor_collections[i].input_tensors[input_index]);
            }
        }
        output_index++;
    }

    std::cout << "Initializing non-pipelined input data..." << std::endl;
    // For all non-pipelined tensors, initialize once
    input_index = 0;
    for (auto& input : tensor_collections[0].input_tensors)
    {
        if (!input->get_stale())
        {
            const shared_ptr<runtime::HostTensor>& data =
                tensor_collections[0].parameter_data[input_index];
            input->write(data->get_data_ptr(),
                         data->get_element_count() * data->get_element_type().size());
        }
        ++input_index;
    }

    std::cout << "Starting threads..." << std::endl;
    thread threads[pipeline_depth];
    for (size_t i = 0; i < pipeline_depth; i++)
    {
        threads[i] = thread(thread_entry, exec.get(), tensor_collections[i], i);
    }

    for (size_t i = 0; i < pipeline_depth; i++)
    {
        threads[i].join();
    }
    float time = s_timer.get_milliseconds();
    cout << time / iterations << "ms per iteration" << endl;

    vector<runtime::PerformanceCounter> perf_data = exec->get_performance_data();
    return perf_data;
}
