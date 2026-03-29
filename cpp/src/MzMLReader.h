#pragma once
#include <string>
#include <vector>
#include <optional>
#include <cstdint>

/**
 * MzMLReader - Reads mzML files (both plain and gzip-compressed .mzML.gz).
 *
 * Parses MS1 and MS2 spectra from mzML files using pugixml.
 * Handles base64-encoded peak data and zlib-compressed arrays.
 */

struct MS1Spectrum {
    double scanTime;          // minutes
    std::vector<double> mz;
    std::vector<double> intensity;
    std::string polarity;     // "positive" or "negative"
    std::string scanId;
    std::string filterString;
};

struct MS2Spectrum {
    double scanTime;          // minutes
    std::vector<double> mz;
    std::vector<double> intensity;
    double precursorMz;
    double precursorIntensity;
    std::string polarity;
    std::string scanId;
};

struct MzMLData {
    std::vector<MS1Spectrum> ms1;
    std::vector<MS2Spectrum> ms2;
};

class MzMLReader {
public:
    MzMLReader() = default;

    // Read an mzML file (or .mzML.gz) and return parsed spectra
    MzMLData readFile(const std::string& filepath) const;

private:
    // Decode a base64 string
    static std::vector<uint8_t> base64Decode(const std::string& encoded);

    // Decompress zlib-compressed data
    static std::vector<uint8_t> zlibDecompress(const std::vector<uint8_t>& compressed);

    // Decode a binary array from mzML (handles base64 + optional zlib + float32/float64)
    static std::vector<double> decodeBinaryArray(const std::string& base64data,
                                                  bool compressed,
                                                  bool is64bit);

    // Read raw bytes from a file (handles .gz decompression)
    static std::string readFileContent(const std::string& filepath);
};
