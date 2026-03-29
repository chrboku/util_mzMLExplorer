#pragma once
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <functional>
#include "MzMLReader.h"
#include "Utils.h"

struct FileEntry {
    std::string filepath;
    std::string filename;
    std::string group;
    std::string color;
    bool loaded = false;
    bool hasError = false;
    std::string errorMessage;
    // Optionally: sample metadata fields
    std::string sampleName;
    std::string sampleType; // e.g. "Sample", "Blank", "QC", "Reference"
    std::string comment;
};

struct LoadedFileData {
    std::vector<MS1Spectrum> ms1;
    std::vector<MS2Spectrum> ms2;
};

class FileManager {
public:
    FileManager();

    // Load files from a TSV/CSV file
    void loadFilesFromTSV(const std::string& tsvPath);

    // Add a single file entry
    void addFile(const FileEntry& entry);

    // Set group colors
    void setGroupColors(const std::map<std::string, std::string>& colors);

    // Get the currently known group colors
    const std::map<std::string, std::string>& getGroupColors() const { return groupColors; }

    // Get all file entries
    const std::vector<FileEntry>& getFiles() const { return files; }

    // Get/set memory mode
    bool isKeepInMemory() const { return keepInMemory; }
    void setMemoryMode(bool keepInMemory, bool autoLoad = true);

    // Load a single file (parses mzML, uses disk cache if available)
    LoadedFileData loadSingleFile(const std::string& filepath);

    // Get cached data for a file (if in memory mode)
    const LoadedFileData* getCachedData(const std::string& filepath) const;

    // Clear all files
    void clearFiles();

    // Clear memory cache
    void clearMemoryCache();

    // Regenerate group colors based on current files
    void regenerateGroupColors();

    // Get the color for a given group
    std::string getGroupColor(const std::string& group) const;

    // Get unique groups in load order
    std::vector<std::string> getUniqueGroups() const;

    // Load all files into memory
    void loadAllFilesToMemory(std::function<void(int, int, const std::string&)> progressCallback = nullptr);

private:
    std::vector<FileEntry> files;
    std::map<std::string, std::string> groupColors;
    bool keepInMemory = false;
    std::map<std::string, LoadedFileData> cachedData;
    MzMLReader reader;

    // Disk cache helpers
    std::string computeFileHash(const std::string& filepath) const;
    std::string getCachePath(const std::string& filepath) const;
    bool loadFromCache(const std::string& filepath, LoadedFileData& out) const;
    void saveToCache(const std::string& filepath, const LoadedFileData& data) const;

    // Parse a TSV/CSV/XLSX-like file
    std::vector<FileEntry> parseTSVFile(const std::string& path) const;
};
