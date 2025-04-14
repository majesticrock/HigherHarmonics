BUILD_DIR = build
CLUSTER_BUILD_DIR = build_cluster
DEBUG_BUILD_DIR = build_debug
NDEBUG_BUILD_DIR = build_ndebug
NO_MPI_BUILD_DIR = build_no_mpi

# Default target to build the project
all: $(BUILD_DIR)/Makefile
	@$(MAKE) -C $(BUILD_DIR)

$(BUILD_DIR)/Makefile: CMakeLists.txt
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCLUSTER_BUILD=OFF ..

ndebug: $(NDEBUG_BUILD_DIR)/Makefile
	@$(MAKE) -C $(NDEBUG_BUILD_DIR)

$(NDEBUG_BUILD_DIR)/Makefile: CMakeLists.txt
	@mkdir -p $(NDEBUG_BUILD_DIR)
	@cd $(NDEBUG_BUILD_DIR) && cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCLUSTER_BUILD=OFF -DCMAKE_BUILD_TYPE=NDEBUG ..

no_mpi: $(NO_MPI_BUILD_DIR)/Makefile
	@$(MAKE) -C $(NO_MPI_BUILD_DIR)

$(NO_MPI_BUILD_DIR)/Makefile: CMakeLists.txt
	@mkdir -p $(NO_MPI_BUILD_DIR)
	@cd $(NO_MPI_BUILD_DIR) && cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCLUSTER_BUILD=OFF -DCMAKE_BUILD_TYPE=NO_MPI ..

# Cluster target to build the project with different compiler flags
cluster: $(CLUSTER_BUILD_DIR)/Makefile
	@$(MAKE) -C $(CLUSTER_BUILD_DIR)

$(CLUSTER_BUILD_DIR)/Makefile: CMakeLists.txt
	@mkdir -p $(CLUSTER_BUILD_DIR)
	@cd $(CLUSTER_BUILD_DIR) && cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCLUSTER_BUILD=ON ..

# Debug target to build the project with debug flags
debug: $(DEBUG_BUILD_DIR)/Makefile
	@$(MAKE) -C $(DEBUG_BUILD_DIR)

$(DEBUG_BUILD_DIR)/Makefile: CMakeLists.txt
	@mkdir -p $(DEBUG_BUILD_DIR)
	@cd $(DEBUG_BUILD_DIR) && cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Debug ..

clean:
	@rm -rf $(BUILD_DIR) $(CLUSTER_BUILD_DIR) $(DEBUG_BUILD_DIR) $(NDEBUG_BUILD_DIR) $(NO_MPI_BUILD_DIR) build_header
	@rm -rf auto_generated*

.PHONY: all clean cluster debug