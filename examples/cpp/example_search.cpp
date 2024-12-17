#include "../../hnswlib/hnswlib.h"


int main() {
    int dim = 128;              // Dimension of the elements
    int max_elements = 1000000; // Maximum number of elements, should be known beforehand
    int M = 64;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 100;  // Controls index search speed/build speed tradeoff
    using T = float;            // Assign a variable T with various types to read data from file
    int num_points, dimension, num_queries, num_elements_per_query;

    const std::string data_file = "/nvmessd2/SIFT1M/sift/sift_base.fbin";
    const std::string query_file = "/nvmessd2/SIFT1M/sift/sift_query.fbin";
    const std::string groundtruth_file = "/nvmessd2/SIFT1M/sift/gt100_base";
    
    // Initiating index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    
    auto data = hnswlib::readDataFromFile<float>(data_file, num_points, dimension);
    auto query_data = hnswlib::readDataFromFile<float>(query_file, num_queries, num_elements_per_query);
    auto groundtruth_data = hnswlib::readGroundTruthFile<int>(groundtruth_file, num_queries, num_elements_per_query);
    
    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    int K = 10; // Number of nearest neighbors to search
    std::vector<float> recall_array(num_queries);
    for (int i = 0; i < num_queries; i++) {
        int correct = 0;
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(query_data + i * dim, K);

        std::unordered_set<int> groundtruth_set;
        for (int j = 0; j < num_elements_per_query; j++) {
            groundtruth_set.insert(groundtruth_data[i * num_elements_per_query + j]);
        }

        while (!result.empty()) {
            if (groundtruth_set.find(result.top().second) != groundtruth_set.end()) {
                correct++;
            }
            result.pop();
        }

        recall_array[i] = (float)correct / K;
    }

    float average_recall = std::accumulate(recall_array.begin(), recall_array.end(), 0.0f) / num_queries;
    std::cout << "Average Recall: " << average_recall << std::endl;
   
    delete[] data, query_data, groundtruth_data;
    return 0;
}
