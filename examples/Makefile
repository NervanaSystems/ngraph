
# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
NGRAPH_DIST_DIR = ${HOME}/ngraph_dist
CFLAGS = -std=c++11
CFLAGS += -I $(NGRAPH_DIST_DIR)
LDFLAGS = -L $(NGRAPH_DIST_DIR)
OBJ = main.o

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CFLAGS)

ngraph-test: $(OBJ) 
	$(CXX) -o $@ $(OBJ) $(LDFLAGS) -lngraph

.PHONY: clean

clean:
	rm -f $(OBJ)