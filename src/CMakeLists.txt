# Copyright 2019-2024 Lawrence Livermore National Security, LLC and other YGM
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

add_ygm_executable(test_sparse)
#add_ygm_executable(shared_mem others/shared_mem.cpp)
add_ygm_executable(test_shm shm_counting_set/test_shm.cpp)
