/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <getopt.h>

#include <iostream>
#include <memory>

#include "ngraph/file_util.hpp"
#include "ngraph/runtime/plaidml/plaidml_backend.hpp"
#include "ngraph/serializer.hpp"

static const struct option opts[] = {{"backend", required_argument, nullptr, 'b'},
                                     {"format", required_argument, nullptr, 'f'},
                                     {"help", no_argument, nullptr, 'h'},
                                     {nullptr, 0, nullptr, '\0'}};

int main(int argc, char** argv)
{
    int opt;
    bool err = false;
    bool usage = false;
    std::string model;
    std::string output;
    std::string backend_name = "PlaidML";
    plaidml_file_format format = PLAIDML_FILE_FORMAT_TILE;

    while ((opt = getopt_long(argc, argv, "f:b:h", opts, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'b': backend_name = optarg; break;
        case 'h': usage = true; break;
        case 'f':
            if (!strcmp(optarg, "tile"))
            {
                format = PLAIDML_FILE_FORMAT_TILE;
            }
            else if (!strcmp(optarg, "human"))
            {
                format = PLAIDML_FILE_FORMAT_STRIPE_HUMAN;
            }
            else if (!strcmp(optarg, "prototxt"))
            {
                format = PLAIDML_FILE_FORMAT_STRIPE_PROTOTXT;
            }
            else if (!strcmp(optarg, "binary"))
            {
                format = PLAIDML_FILE_FORMAT_STRIPE_BINARY;
            }
            else
            {
                err = true;
            }
            break;
        case '?':
        default: err = true; break;
        }
    }

    if (optind + 2 != argc)
    {
        err = true;
    }
    else
    {
        model = argv[optind];
        output = argv[optind + 1];

        if (model.empty())
        {
            err = true;
        }
        else if (!ngraph::file_util::exists(model))
        {
            std::cerr << "File " << model << " not found\n";
            err = true;
        }

        if (output.empty())
        {
            err = true;
        }
        else if (ngraph::file_util::exists(output))
        {
            std::cerr << "File " << output << " already exists; not overwriting\n";
            err = true;
        }
    }

    if (backend_name.substr(0, backend_name.find(':')) != "PlaidML")
    {
        std::cerr << "Unsupported backend: " << backend_name << "\n";
        err = true;
    }

    if (err || usage)
    {
        std::cerr << R"###(
DESCRIPTION
       Convert an ngraph JSON model to one of PlaidML's file formats.

SYNOPSIS
       ngraph-to-plaidml [--backend|-b <backend>] MODEL OUTPUT

OPTIONS
        -b|--backend      Backend to use (default: PlaidML)
        -f|--format       Format to use (tile, human, prototxt, binary, or json; default: tile)
)###";
    }
    if (err)
    {
        return EXIT_FAILURE;
    }
    if (usage)
    {
        return EXIT_SUCCESS;
    }

    std::cerr << "Reading nGraph model from " << model << "\n";
    std::shared_ptr<ngraph::Function> f = ngraph::deserialize(model);
    std::shared_ptr<ngraph::runtime::Backend> base_backend =
        ngraph::runtime::Backend::create(backend_name);
    std::shared_ptr<ngraph::runtime::plaidml::PlaidML_Backend> backend =
        std::dynamic_pointer_cast<ngraph::runtime::plaidml::PlaidML_Backend>(base_backend);
    if (!backend)
    {
        std::cerr << "Failed to load PlaidML backend\n";
        return EXIT_FAILURE;
    }

    backend->save(f, output, format);
    std::cerr << "Wrote output to " << output << "\n";
    return EXIT_SUCCESS;
}
