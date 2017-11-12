/*
 Copyright 2016 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include <cassert>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <ftw.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "file_util.hpp"

using namespace std;

string ngraph::file_util::path_join(const string& s1, const string& s2)
{
    string rc;
    if (s2.size() > 0)
    {
        if (s2[0] == '/')
        {
            rc = s2;
        }
        else if (s1.size() > 0)
        {
            rc = s1;
            if (rc[rc.size() - 1] != '/')
            {
                rc += "/";
            }
            rc += s2;
        }
        else
        {
            rc = s2;
        }
    }
    else
    {
        rc = s1;
    }
    return rc;
}

size_t ngraph::file_util::get_file_size(const string& filename)
{
    // ensure that filename exists and get its size

    struct stat stats;
    if (stat(filename.c_str(), &stats) == -1)
    {
        throw std::runtime_error("Could not find file: \"" + filename + "\"");
    }

    return stats.st_size;
}

void ngraph::file_util::remove_directory(const string& dir)
{
    file_util::iterate_files(dir,
                             [](const string& file, bool is_dir) {
                                 if (is_dir)
                                     rmdir(file.c_str());
                                 else
                                     remove(file.c_str());
                             },
                             true);
    rmdir(dir.c_str());
}

void ngraph::file_util::remove_file(const string& file)
{
    remove(file.c_str());
}

bool ngraph::file_util::make_directory(const string& dir)
{
    if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
    {
        if (errno == EEXIST)
        {
            // not really an error, the directory already exists
            return false;
        }
        throw std::runtime_error("error making directory " + dir + " " + strerror(errno));
    }
    return true;
}

string ngraph::file_util::make_temp_directory(const string& path)
{
    string fname = path.empty() ? file_util::get_temp_directory() : path;
    string tmp_template = file_util::path_join(fname, "aeonXXXXXX");
    char* tmpname = strdup(tmp_template.c_str());

    mkdtemp(tmpname);

    string rc = tmpname;
    free(tmpname);
    return rc;
}

std::string ngraph::file_util::get_temp_directory()
{
    const vector<string> potential_tmps = {"NERVANA_AEON_TMP", "TMPDIR", "TMP", "TEMP", "TEMPDIR"};

    const char* path = nullptr;
    for (const string& var : potential_tmps)
    {
        path = getenv(var.c_str());
        if (path != nullptr)
        {
            break;
        }
    }
    if (path == nullptr)
    {
        path = "/tmp";
    }

    return path;
}

vector<char> ngraph::file_util::read_file_contents(const string& path)
{
    size_t file_size = get_file_size(path);
    vector<char> data;
    data.reserve(file_size);
    data.resize(file_size);

    FILE* f = fopen(path.c_str(), "rb");
    if (f)
    {
        char* p = data.data();
        size_t remainder = file_size;
        size_t offset = 0;
        while (remainder > 0)
        {
            size_t rc = fread(&p[offset], 1, remainder, f);
            offset += rc;
            remainder -= rc;
        }
        fclose(f);
    }
    else
    {
        throw std::runtime_error("error opening file '" + path + "'");
    }
    return data;
}

std::string ngraph::file_util::read_file_to_string(const std::string& path)
{
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void ngraph::file_util::iterate_files(const string& path,
                                      std::function<void(const string& file, bool is_dir)> func,
                                      bool recurse)
{
    vector<string> files;
    vector<string> dirs;
    file_util::iterate_files_worker(path,
                                    [&files, &dirs](const string& file, bool is_dir) {
                                        if (is_dir)
                                            dirs.push_back(file);
                                        else
                                            files.push_back(file);
                                    },
                                    true);

    for (auto f : files)
        func(f, false);
    for (auto f : dirs)
        func(f, true);
}

void ngraph::file_util::iterate_files_worker(
    const string& path, std::function<void(const string& file, bool is_dir)> func, bool recurse)
{
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(path.c_str())) != nullptr)
    {
        while ((ent = readdir(dir)) != nullptr)
        {
            string name = ent->d_name;
            switch (ent->d_type)
            {
            case DT_DIR:
                if (recurse && name != "." && name != "..")
                {
                    string dir_path = file_util::path_join(path, name);
                    iterate_files(dir_path, func, recurse);
                    func(dir_path, true);
                }
                break;
            case DT_LNK: break;
            case DT_REG:
                name = file_util::path_join(path, name);
                func(name, false);
                break;
            default: break;
            }
        }
        closedir(dir);
    }
    else
    {
        throw std::runtime_error("error enumerating file " + path);
    }
}

string ngraph::file_util::tmp_filename(const string& extension)
{
    string tmp_template =
        file_util::path_join(file_util::get_temp_directory(), "ngraph_XXXXXX" + extension);
    char* tmpname = strdup(tmp_template.c_str());

    // mkstemp opens the file with open() so we need to close it
    close(mkstemps(tmpname, static_cast<int>(extension.size())));

    string rc = tmpname;
    free(tmpname);
    return rc;
}

void ngraph::file_util::touch(const std::string& filename)
{
    // inspired by http://chris-sharpe.blogspot.com/2013/05/better-than-systemtouch.html
    int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_NOCTTY | O_NONBLOCK, 0666);
    assert(fd >= 0);
    close(fd);

    // update timestamp for filename
    int rc = utimes(filename.c_str(), nullptr);
    assert(!rc);
}

bool ngraph::file_util::exists(const std::string& filename)
{
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

int ngraph::file_util::try_get_lock(const std::string& filename)
{
    mode_t m = umask(0);
    int fd = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
    umask(m);
    if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) < 0)
    {
        close(fd);
        fd = -1;
    }
    return fd;
}

void ngraph::file_util::release_lock(int fd, const std::string& filename)
{
    if (fd >= 0)
    {
        remove_file(filename);
        close(fd);
    }
}
