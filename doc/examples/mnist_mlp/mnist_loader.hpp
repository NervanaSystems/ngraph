//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

class MNistLoader
{
protected:
    MNistLoader(const std::string& filename, std::uint32_t magic);
    virtual ~MNistLoader();
    virtual void read_header();

public:
    void open();
    void close();
    void reset();

    template <typename T>
    void read(T* loc, size_t n = 1);

    void read_scaled(float* loc, size_t n);

    template <typename T>
    size_t file_read(T* loc, size_t n)
    {
        return fread(loc, sizeof(T), n, m_file);
    }

    std::uint32_t get_items() { return m_items; }
protected:
    std::string m_filename;
    FILE* m_file{nullptr};
    std::uint32_t m_magic;
    std::uint32_t m_items;
    fpos_t m_data_pos;
};

class MNistImageLoader : public MNistLoader
{
    static const std::uint32_t magic_value = 0x00000803;

    virtual void read_header() override;

public:
    MNistImageLoader(const std::string& file);

    static const char* const TEST;
    static const char* const TRAIN;

    std::uint32_t get_rows() { return m_rows; }
    std::uint32_t get_columns() { return m_columns; }
protected:
    std::uint32_t m_rows;
    std::uint32_t m_columns;
};

class MNistLabelLoader : public MNistLoader
{
    static const std::uint32_t magic_value = 0x00000801;

public:
    MNistLabelLoader(const std::string& file);

    static const char* TEST;
    static const char* TRAIN;
};

class MNistDataLoader
{
public:
    MNistDataLoader(size_t batch_size,
                    const std::string& image,
                    const std::string& label);
    ~MNistDataLoader();

    void open();
    void close();

    std::uint32_t get_rows() { return m_image_loader.get_rows(); }
    std::uint32_t get_columns() { return m_image_loader.get_columns(); }
    size_t get_batch_size() { return m_batch_size; }
    size_t get_items() { return m_items; }
    size_t get_epoch() { return m_epoch; }
    size_t get_pos() { return m_pos; }
    void load();
    void rewind();
    void reset();

    const float* get_image_floats() const { return m_image_floats.get(); }
    const float* get_label_floats() const { return m_label_floats.get(); }
    size_t get_image_batch_size() const
    {
        return m_image_sample_size * m_batch_size;
    }
    size_t get_label_batch_size() const { return m_batch_size; }
protected:
    size_t m_batch_size;
    MNistImageLoader m_image_loader;
    MNistLabelLoader m_label_loader;
    std::int32_t m_items;
    size_t m_pos{0};
    size_t m_epoch{0};
    std::unique_ptr<float[]> m_image_floats;
    std::unique_ptr<float[]> m_label_floats;
    size_t m_image_sample_size;
};
