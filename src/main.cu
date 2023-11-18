/**
    CudAD is a CUDA neural network trainer, specific for chess engines.
    Copyright (C) 2022 Finn Eggers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "archs/Koivisto.h"
#include "archs/Clover.h"
//#include "archs/Alexandria.h"
#include "misc/config.h"
#include "numerical/finite_difference.h"
#include "trainer.h"
#include "quantitize.h"
#include "dataset/writer.h"
#include "dataset/batchloader.h"
#include "dataset/dataset.h"
#include "dataset/shuffle.h"

#include <iostream>
#include <vector>

using namespace std;

const bool loadData = false;
const bool shuffleData = false;
const bool trainData = true;

struct RootPath {
    string path;
    int init_ind;
};

int main() {
    init();

    vector <RootPath> rootPaths {{"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_3_2_v2_", 0},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_6_", 16},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\Clover_data_5k_3_", 48},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\Clover_data_5k_2_", 64},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\Clover_data_5k_", 80},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\Clover_4_0_data_", 96},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\Clover_4.0_data_", 112},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\Clover_3_3_5knodes_", 128},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_3_2_v3_", 144},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_3_2_v1_", 160},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_10_", 176},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_5_", 192},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_3_3_v2_", 208},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_5k_", 224},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_5k_2_", 240},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_5k_3_", 256},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_1_", 272},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_2_", 288},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_3_", 304},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_4_", 320},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_7_", 336},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_8_", 352},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_9_", 368},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_11_", 384},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_12_", 400},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_13_", 416},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_14_", 432},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_15_", 448},
                               {"C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_16_", 464},
                               };

    DataSet CloverData{}, ds{};
    vector <string> init_files{};
    if(loadData) {
        int ind = 0;
        for(auto &[rootPath, start_ind] : rootPaths) {
            ind = start_ind;
            for(int i = 0; i < 16; i++) {
                DataSet ds = read<TEXT>(rootPath + to_string(i) + ".txt");
                //ds.shuffle();
                write("C:\\Users\\Luca\\source\\repos\\CudAD\\data\\CloverData_test" + to_string(ind) + ".bin", ds, (int)1e8);
                ind++;
            }
        }
    }

    if(false) {
        int ind = 464;
        for(int i = 0; i < 16; i++) {
            DataSet ds = read<TEXT>("C:\\Users\\Luca\\Desktop\\CloverData\\CloverData_16_" + to_string(i) + ".txt");
            //ds.shuffle();
            write("C:\\Users\\Luca\\source\\repos\\CudAD\\data\\CloverData_test" + to_string(ind) + ".bin", ds, (int)1e8);
            ind++;
        }
    }

    if(false) {
        int ind = 0;
        for(auto &[rootPath, start_ind] : rootPaths) {
            //bool flag = (ind / 16 == 12 || ind / 16 == 14);
            ind = start_ind;
            for(int i = 0; i < 16; i++) {
                ind++;
                init_files.push_back("C:\\Users\\Luca\\source\\repos\\CudAD\\data\\CloverData_test" + to_string(ind) + ".bin");
            }
        }
        mix_and_shuffle_2(init_files, "C:\\Users\\Luca\\source\\repos\\CudAD\\data\\Clover_shuffled$.bin");
    }

    vector<string> files {};
    const string   output    = R"(C:\Users\Luca\source\repos\CudAD\runs\run23\)";
    
    for (int i = 1; i <= 32; i++)
        files.push_back("C:\\Users\\Luca\\source\\repos\\CudAD\\data\\Clover_shuffled" + to_string(i) + ".bin");

    std::cout << "Pushed all files\n";
    if(trainData) {
        Trainer<Clover> trainer {};
        trainer.fit(
            files,
            vector<string> {"C:\\Users\\Luca\\source\\repos\\CudAD\\data\\CloverData_test69.bin"},
            output);
    }

    auto layers = Clover::get_layers();
    Network network{std::get<0>(layers),std::get<1>(layers)};
    network.setLossFunction(Clover::get_loss_function());
    network.loadWeights(output + "weights-5buckets-epoch350.nnue");
    
    test_fen<Clover>(network, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    test_fen<Clover>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
    test_fen<Clover>(network, "3k4/8/8/2QK4/8/8/8/8 w - - 0 1");
    test_fen<Clover>(network, "3k4/8/8/2RK4/8/8/8/8 w - - 0 1");
    test_fen<Clover>(network, "2k5/8/1P6/2KN3P/8/8/8/8 w - - 1 71");
////    network.loadWeights(output + "weights-epoch10.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch20.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch30.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch40.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch50.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch60.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch70.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch80.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch90.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch100.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch110.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch120.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch130.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch140.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch160.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch420.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////
////    finite_difference(network, target, target_mask);
//
////    network.loadWeights(output + "weights-epoch120.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////    network.loadWeights(output + "weights-epoch30.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////
////    network.loadWeights(output + "weights-epoch50.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
////
////
////    network.loadWeights(output + "weights-epoch80.nnue");
////    test_fen<Koivisto>(network, "8/8/6R1/5k1P/6p1/4K3/8/8 b - - 1 53");
//
////    test_fen<Koivisto>(network, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
////    test_fen<Koivisto>(network, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
//
//
    BatchLoader batch_loader{files, 16384};
    batch_loader.start();
    computeScalars<Clover>(batch_loader, network, 128);
//
    //auto f = openFile(output + "Clover.net");
    //writeLayer<int16_t, int16_t>(f, network, 0, 32, 32);
    //writeLayer<int16_t, int32_t>(f, network, 4, 256, 256);
    //closeFile(f);

    close();
}
