#include "FileManager.h"
#include "Utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <openssl/sha.h>
#include <iomanip>

namespace fs = std::filesystem;

// Simple SHA-256 implementation using system libraries
static std::string sha256File(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) return "";

    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    char buf[65536];
    while (ifs.read(buf, sizeof(buf)) || ifs.gcount() > 0) {
        SHA256_Update(&ctx, buf, (size_t)ifs.gcount());
    }

    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_Final(hash, &ctx);

    std::ostringstream oss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i)
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    return oss.str();
}

FileManager::FileManager() {}

void FileManager::clearFiles() {
    files.clear();
    groupColors.clear();
    cachedData.clear();
}

void FileManager::clearMemoryCache() {
    cachedData.clear();
}

std::string FileManager::computeFileHash(const std::string& filepath) const {
    return sha256File(filepath);
}

std::string FileManager::getCachePath(const std::string& filepath) const {
    return filepath + ".cached.bin";
}

bool FileManager::loadFromCache(const std::string& filepath, LoadedFileData& out) const {
    std::string cachePath = getCachePath(filepath);
    if (!fs::exists(cachePath)) return false;

    try {
        std::string currentHash = computeFileHash(filepath);

        std::ifstream ifs(cachePath, std::ios::binary);
        if (!ifs) return false;

        // Read stored hash
        size_t hashLen;
        ifs.read(reinterpret_cast<char*>(&hashLen), sizeof(hashLen));
        std::string storedHash(hashLen, '\0');
        ifs.read(&storedHash[0], hashLen);

        if (storedHash != currentHash) {
            fs::remove(cachePath);
            return false;
        }

        // Read MS1 spectra
        size_t ms1Count;
        ifs.read(reinterpret_cast<char*>(&ms1Count), sizeof(ms1Count));
        out.ms1.resize(ms1Count);
        for (auto& sp : out.ms1) {
            ifs.read(reinterpret_cast<char*>(&sp.scanTime), sizeof(sp.scanTime));

            size_t mzSize;
            ifs.read(reinterpret_cast<char*>(&mzSize), sizeof(mzSize));
            sp.mz.resize(mzSize);
            sp.intensity.resize(mzSize);
            if (mzSize > 0) {
                ifs.read(reinterpret_cast<char*>(sp.mz.data()), mzSize * sizeof(double));
                ifs.read(reinterpret_cast<char*>(sp.intensity.data()), mzSize * sizeof(double));
            }

            size_t polLen;
            ifs.read(reinterpret_cast<char*>(&polLen), sizeof(polLen));
            sp.polarity.resize(polLen);
            if (polLen > 0) ifs.read(&sp.polarity[0], polLen);

            size_t scanIdLen;
            ifs.read(reinterpret_cast<char*>(&scanIdLen), sizeof(scanIdLen));
            sp.scanId.resize(scanIdLen);
            if (scanIdLen > 0) ifs.read(&sp.scanId[0], scanIdLen);

            size_t fsLen;
            ifs.read(reinterpret_cast<char*>(&fsLen), sizeof(fsLen));
            sp.filterString.resize(fsLen);
            if (fsLen > 0) ifs.read(&sp.filterString[0], fsLen);
        }

        // Read MS2 spectra
        size_t ms2Count;
        ifs.read(reinterpret_cast<char*>(&ms2Count), sizeof(ms2Count));
        out.ms2.resize(ms2Count);
        for (auto& sp : out.ms2) {
            ifs.read(reinterpret_cast<char*>(&sp.scanTime), sizeof(sp.scanTime));
            ifs.read(reinterpret_cast<char*>(&sp.precursorMz), sizeof(sp.precursorMz));
            ifs.read(reinterpret_cast<char*>(&sp.precursorIntensity), sizeof(sp.precursorIntensity));

            size_t mzSize;
            ifs.read(reinterpret_cast<char*>(&mzSize), sizeof(mzSize));
            sp.mz.resize(mzSize);
            sp.intensity.resize(mzSize);
            if (mzSize > 0) {
                ifs.read(reinterpret_cast<char*>(sp.mz.data()), mzSize * sizeof(double));
                ifs.read(reinterpret_cast<char*>(sp.intensity.data()), mzSize * sizeof(double));
            }

            size_t polLen;
            ifs.read(reinterpret_cast<char*>(&polLen), sizeof(polLen));
            sp.polarity.resize(polLen);
            if (polLen > 0) ifs.read(&sp.polarity[0], polLen);

            size_t scanIdLen;
            ifs.read(reinterpret_cast<char*>(&scanIdLen), sizeof(scanIdLen));
            sp.scanId.resize(scanIdLen);
            if (scanIdLen > 0) ifs.read(&sp.scanId[0], scanIdLen);
        }

        return ifs.good();
    } catch (...) {
        return false;
    }
}

void FileManager::saveToCache(const std::string& filepath, const LoadedFileData& data) const {
    std::string cachePath = getCachePath(filepath);
    try {
        std::string hash = computeFileHash(filepath);
        std::ofstream ofs(cachePath, std::ios::binary);
        if (!ofs) return;

        // Write hash
        size_t hashLen = hash.size();
        ofs.write(reinterpret_cast<const char*>(&hashLen), sizeof(hashLen));
        ofs.write(hash.data(), hashLen);

        // Write MS1
        size_t ms1Count = data.ms1.size();
        ofs.write(reinterpret_cast<const char*>(&ms1Count), sizeof(ms1Count));
        for (const auto& sp : data.ms1) {
            ofs.write(reinterpret_cast<const char*>(&sp.scanTime), sizeof(sp.scanTime));
            size_t mzSize = sp.mz.size();
            ofs.write(reinterpret_cast<const char*>(&mzSize), sizeof(mzSize));
            if (mzSize > 0) {
                ofs.write(reinterpret_cast<const char*>(sp.mz.data()), mzSize * sizeof(double));
                ofs.write(reinterpret_cast<const char*>(sp.intensity.data()), mzSize * sizeof(double));
            }
            size_t polLen = sp.polarity.size();
            ofs.write(reinterpret_cast<const char*>(&polLen), sizeof(polLen));
            ofs.write(sp.polarity.data(), polLen);
            size_t scanIdLen = sp.scanId.size();
            ofs.write(reinterpret_cast<const char*>(&scanIdLen), sizeof(scanIdLen));
            ofs.write(sp.scanId.data(), scanIdLen);
            size_t fsLen = sp.filterString.size();
            ofs.write(reinterpret_cast<const char*>(&fsLen), sizeof(fsLen));
            ofs.write(sp.filterString.data(), fsLen);
        }

        // Write MS2
        size_t ms2Count = data.ms2.size();
        ofs.write(reinterpret_cast<const char*>(&ms2Count), sizeof(ms2Count));
        for (const auto& sp : data.ms2) {
            ofs.write(reinterpret_cast<const char*>(&sp.scanTime), sizeof(sp.scanTime));
            ofs.write(reinterpret_cast<const char*>(&sp.precursorMz), sizeof(sp.precursorMz));
            ofs.write(reinterpret_cast<const char*>(&sp.precursorIntensity), sizeof(sp.precursorIntensity));
            size_t mzSize = sp.mz.size();
            ofs.write(reinterpret_cast<const char*>(&mzSize), sizeof(mzSize));
            if (mzSize > 0) {
                ofs.write(reinterpret_cast<const char*>(sp.mz.data()), mzSize * sizeof(double));
                ofs.write(reinterpret_cast<const char*>(sp.intensity.data()), mzSize * sizeof(double));
            }
            size_t polLen = sp.polarity.size();
            ofs.write(reinterpret_cast<const char*>(&polLen), sizeof(polLen));
            ofs.write(sp.polarity.data(), polLen);
            size_t scanIdLen = sp.scanId.size();
            ofs.write(reinterpret_cast<const char*>(&scanIdLen), sizeof(scanIdLen));
            ofs.write(sp.scanId.data(), scanIdLen);
        }
    } catch (...) {
        // Cache save failure is non-fatal
    }
}

LoadedFileData FileManager::loadSingleFile(const std::string& filepath) {
    LoadedFileData data;

    // Try disk cache
    if (loadFromCache(filepath, data)) {
        std::cout << "Loaded from cache: " << fs::path(filepath).filename().string() << "\n";
        return data;
    }

    std::cout << "Loading: " << fs::path(filepath).filename().string() << "...\n";
    auto mzmlData = reader.readFile(filepath);
    data.ms1 = std::move(mzmlData.ms1);
    data.ms2 = std::move(mzmlData.ms2);

    saveToCache(filepath, data);
    return data;
}

const LoadedFileData* FileManager::getCachedData(const std::string& filepath) const {
    auto it = cachedData.find(filepath);
    if (it != cachedData.end()) return &it->second;
    return nullptr;
}

void FileManager::addFile(const FileEntry& entry) {
    files.push_back(entry);
}

void FileManager::setGroupColors(const std::map<std::string, std::string>& colors) {
    groupColors = colors;
}

void FileManager::setMemoryMode(bool keep, bool autoLoad) {
    if (keepInMemory == keep) return;
    keepInMemory = keep;

    if (keep && autoLoad) {
        loadAllFilesToMemory();
    } else if (!keep) {
        clearMemoryCache();
    }
}

void FileManager::loadAllFilesToMemory(std::function<void(int, int, const std::string&)> progress) {
    int total = (int)files.size();
    for (int i = 0; i < total; ++i) {
        if (progress) progress(i, total, files[i].filepath);
        if (files[i].loaded && !files[i].hasError) {
            if (cachedData.find(files[i].filepath) == cachedData.end()) {
                try {
                    cachedData[files[i].filepath] = loadSingleFile(files[i].filepath);
                } catch (const std::exception& e) {
                    std::cerr << "Error loading " << files[i].filepath << ": " << e.what() << "\n";
                }
            }
        }
    }
}

void FileManager::regenerateGroupColors() {
    auto groups = getUniqueGroups();
    auto palette = Utils::generateColorPalette((int)groups.size());
    groupColors.clear();
    for (int i = 0; i < (int)groups.size(); ++i) {
        groupColors[groups[i]] = palette[i];
    }
}

std::string FileManager::getGroupColor(const std::string& group) const {
    auto it = groupColors.find(group);
    if (it != groupColors.end()) return it->second;
    return "#1f77b4"; // default blue
}

std::vector<std::string> FileManager::getUniqueGroups() const {
    std::vector<std::string> groups;
    for (const auto& f : files) {
        if (std::find(groups.begin(), groups.end(), f.group) == groups.end()) {
            groups.push_back(f.group);
        }
    }
    return groups;
}

std::vector<FileEntry> FileManager::parseTSVFile(const std::string& path) const {
    std::vector<FileEntry> entries;
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Cannot open file: " + path);

    std::string line;
    std::getline(ifs, line);

    // Detect delimiter
    char delim = '\t';
    if (line.find(',') != std::string::npos && line.find('\t') == std::string::npos) {
        delim = ',';
    }

    // Parse header
    std::vector<std::string> headers;
    {
        std::istringstream ss(line);
        std::string field;
        while (std::getline(ss, field, delim)) {
            // Remove BOM and trim
            if (!field.empty() && field[0] == '\xef') field = field.substr(3);
            while (!field.empty() && (field.back() == '\r' || field.back() == '\n' || field.back() == ' '))
                field.pop_back();
            std::transform(field.begin(), field.end(), field.begin(), ::tolower);
            headers.push_back(field);
        }
    }

    auto colIdx = [&](const std::string& name) -> int {
        for (int i = 0; i < (int)headers.size(); ++i)
            if (headers[i] == name) return i;
        return -1;
    };

    int filepathCol = colIdx("filepath");
    int filenameCol = colIdx("filename");
    int groupCol    = colIdx("group");
    int sampleCol   = colIdx("samplename");
    int typeCol     = colIdx("sampletype");
    int commentCol  = colIdx("comment");

    if (filepathCol < 0 && filenameCol < 0) {
        filepathCol = 0; // assume first column is filepath
    }

    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::vector<std::string> fields;
        {
            std::istringstream ss(line);
            std::string field;
            while (std::getline(ss, field, delim)) {
                while (!field.empty() && (field.back() == '\r' || field.back() == '\n'))
                    field.pop_back();
                fields.push_back(field);
            }
        }

        auto getField = [&](int idx) -> std::string {
            if (idx >= 0 && idx < (int)fields.size()) return fields[idx];
            return "";
        };

        FileEntry e;
        if (filepathCol >= 0) e.filepath = getField(filepathCol);
        else if (filenameCol >= 0) e.filepath = getField(filenameCol);

        if (e.filepath.empty()) continue;

        // Extract filename from path if not provided
        e.filename = fs::path(e.filepath).filename().string();
        if (filenameCol >= 0) {
            std::string fn = getField(filenameCol);
            if (!fn.empty()) e.filename = fn;
        }

        e.group      = groupCol >= 0 ? getField(groupCol) : "Default";
        e.sampleName = sampleCol >= 0 ? getField(sampleCol) : "";
        e.sampleType = typeCol >= 0 ? getField(typeCol) : "";
        e.comment    = commentCol >= 0 ? getField(commentCol) : "";

        entries.push_back(e);
    }

    return entries;
}

void FileManager::loadFilesFromTSV(const std::string& tsvPath) {
    auto newEntries = parseTSVFile(tsvPath);
    for (auto& e : newEntries) {
        files.push_back(e);
    }
    regenerateGroupColors();
}
