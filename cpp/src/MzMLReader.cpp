#include "MzMLReader.h"
#include "pugixml.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <zlib.h>

// ---------------------------------------------------------------------------
// Base64 decode
// ---------------------------------------------------------------------------
static const std::string B64_CHARS =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::vector<uint8_t> MzMLReader::base64Decode(const std::string& encoded) {
    std::vector<uint8_t> decoded;
    int i = 0;
    unsigned char charArray4[4], charArray3[3];
    int idx = 0;

    for (char c : encoded) {
        if (c == '=') break;
        if (!std::isalnum(c) && c != '+' && c != '/') continue;

        charArray4[idx++] = (unsigned char)c;
        if (idx == 4) {
            for (int j = 0; j < 4; j++)
                charArray4[j] = (unsigned char)B64_CHARS.find(charArray4[j]);

            charArray3[0] = (charArray4[0] << 2) | (charArray4[1] >> 4);
            charArray3[1] = ((charArray4[1] & 0xf) << 4) | (charArray4[2] >> 2);
            charArray3[2] = ((charArray4[2] & 0x3) << 6) | charArray4[3];

            for (int j = 0; j < 3; j++) decoded.push_back(charArray3[j]);
            idx = 0;
        }
    }

    if (idx > 0) {
        for (int j = idx; j < 4; j++) charArray4[j] = 0;
        for (int j = 0; j < 4; j++)
            charArray4[j] = (unsigned char)B64_CHARS.find(charArray4[j]);

        charArray3[0] = (charArray4[0] << 2) | (charArray4[1] >> 4);
        charArray3[1] = ((charArray4[1] & 0xf) << 4) | (charArray4[2] >> 2);
        charArray3[2] = ((charArray4[2] & 0x3) << 6) | charArray4[3];

        for (int j = 0; j < idx - 1; j++) decoded.push_back(charArray3[j]);
    }

    return decoded;
}

// ---------------------------------------------------------------------------
// Zlib decompress
// ---------------------------------------------------------------------------
std::vector<uint8_t> MzMLReader::zlibDecompress(const std::vector<uint8_t>& compressed) {
    std::vector<uint8_t> decompressed;
    decompressed.resize(compressed.size() * 4);

    z_stream zs{};
    if (inflateInit(&zs) != Z_OK)
        throw std::runtime_error("inflateInit failed");

    zs.next_in  = const_cast<Bytef*>(compressed.data());
    zs.avail_in = (uInt)compressed.size();

    int ret;
    do {
        if (zs.total_out >= decompressed.size())
            decompressed.resize(decompressed.size() * 2);

        zs.next_out  = decompressed.data() + zs.total_out;
        zs.avail_out = (uInt)(decompressed.size() - zs.total_out);

        ret = inflate(&zs, Z_NO_FLUSH);
    } while (ret == Z_OK);

    inflateEnd(&zs);

    if (ret != Z_STREAM_END)
        throw std::runtime_error("zlib inflate failed");

    decompressed.resize(zs.total_out);
    return decompressed;
}

// ---------------------------------------------------------------------------
// Decode binary array
// ---------------------------------------------------------------------------
std::vector<double> MzMLReader::decodeBinaryArray(const std::string& base64data,
                                                    bool compressed,
                                                    bool is64bit) {
    auto bytes = base64Decode(base64data);

    if (compressed) {
        bytes = zlibDecompress(bytes);
    }

    std::vector<double> result;
    if (is64bit) {
        size_t n = bytes.size() / 8;
        result.resize(n);
        for (size_t i = 0; i < n; ++i) {
            double val;
            std::memcpy(&val, bytes.data() + i * 8, 8);
            result[i] = val;
        }
    } else {
        size_t n = bytes.size() / 4;
        result.resize(n);
        for (size_t i = 0; i < n; ++i) {
            float val;
            std::memcpy(&val, bytes.data() + i * 4, 4);
            result[i] = (double)val;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Read file content (with gz support)
// ---------------------------------------------------------------------------
std::string MzMLReader::readFileContent(const std::string& filepath) {
    // Check for .gz extension
    bool isGzip = filepath.size() >= 3 &&
                  filepath.substr(filepath.size() - 3) == ".gz";

    if (isGzip) {
        gzFile gf = gzopen(filepath.c_str(), "rb");
        if (!gf) throw std::runtime_error("Cannot open gzip file: " + filepath);

        std::string content;
        char buf[65536];
        int bytesRead;
        while ((bytesRead = gzread(gf, buf, sizeof(buf))) > 0) {
            content.append(buf, bytesRead);
        }
        gzclose(gf);
        return content;
    } else {
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs) throw std::runtime_error("Cannot open file: " + filepath);
        return std::string(std::istreambuf_iterator<char>(ifs),
                           std::istreambuf_iterator<char>());
    }
}

// ---------------------------------------------------------------------------
// Main parse function
// ---------------------------------------------------------------------------
MzMLData MzMLReader::readFile(const std::string& filepath) const {
    MzMLData result;

    std::string content = readFileContent(filepath);

    pugi::xml_document doc;
    pugi::xml_parse_result parseResult = doc.load_string(content.c_str());
    if (!parseResult) {
        throw std::runtime_error("Failed to parse mzML: " + std::string(parseResult.description()));
    }

    // Find the spectrumList node
    auto root = doc.first_child(); // indexedmzML or mzML
    if (std::string(root.name()) == "indexedmzML") {
        root = root.child("mzML");
    }

    auto run = root.child("run");
    auto spectrumList = run.child("spectrumList");

    for (auto spectrum : spectrumList.children("spectrum")) {
        // Determine MS level
        int msLevel = 0;
        std::string polarity;
        std::string scanId = spectrum.attribute("id").as_string();
        double scanTime = 0.0;
        std::string filterString;

        // Parse cvParams from spectrum
        for (auto cvp : spectrum.children("cvParam")) {
            std::string acc = cvp.attribute("accession").as_string();
            if (acc == "MS:1000511") { // ms level
                msLevel = cvp.attribute("value").as_int();
            } else if (acc == "MS:1000130") { // positive scan
                polarity = "positive";
            } else if (acc == "MS:1000129") { // negative scan
                polarity = "negative";
            } else if (acc == "MS:1000512") { // filter string
                filterString = cvp.attribute("value").as_string();
            }
        }

        // Get scan time from scanList
        auto scanList = spectrum.child("scanList");
        for (auto scan : scanList.children("scan")) {
            for (auto cvp : scan.children("cvParam")) {
                std::string acc = cvp.attribute("accession").as_string();
                if (acc == "MS:1000016") { // scan start time
                    scanTime = cvp.attribute("value").as_double();
                    // Check unit - if minutes, it's already in minutes
                    // unitAccession MS:1000038 = minute, MS:1000031 = second
                    std::string unitAcc = cvp.attribute("unitAccession").as_string();
                    if (unitAcc == "UO:0000010" || unitAcc == "MS:1000031") {
                        scanTime /= 60.0; // convert seconds to minutes
                    }
                }
            }
            // Also check userParams for filter string
            for (auto userp : scan.children("userParam")) {
                std::string name = userp.attribute("name").as_string();
                if (name == "filter string" || name == "[Thermo Trailer Extra]Filter String:") {
                    filterString = userp.attribute("value").as_string();
                }
            }
        }

        // Parse binary data arrays
        std::vector<double> mzArray, intensityArray;
        auto binaryDataArrayList = spectrum.child("binaryDataArrayList");
        for (auto bda : binaryDataArrayList.children("binaryDataArray")) {
            bool compressed = false;
            bool is64bit = true;
            bool isMz = false;
            bool isIntensity = false;

            for (auto cvp : bda.children("cvParam")) {
                std::string acc = cvp.attribute("accession").as_string();
                if (acc == "MS:1000574") compressed = true;     // zlib compression
                else if (acc == "MS:1000514") isMz = true;      // m/z array
                else if (acc == "MS:1000515") isIntensity = true; // intensity array
                else if (acc == "MS:1000521") is64bit = false;  // 32-bit float
                else if (acc == "MS:1000523") is64bit = true;   // 64-bit float
            }

            std::string b64data = bda.child_value("binary");
            if (!b64data.empty()) {
                auto decoded = decodeBinaryArray(b64data, compressed, is64bit);
                if (isMz) mzArray = decoded;
                else if (isIntensity) intensityArray = decoded;
            }
        }

        if (msLevel == 1) {
            MS1Spectrum sp;
            sp.scanTime    = scanTime;
            sp.mz          = std::move(mzArray);
            sp.intensity   = std::move(intensityArray);
            sp.polarity    = polarity;
            sp.scanId      = scanId;
            sp.filterString = filterString;
            result.ms1.push_back(std::move(sp));

        } else if (msLevel == 2) {
            // Get precursor info
            double precursorMz = 0.0;
            double precursorIntensity = 0.0;

            auto precursorList = spectrum.child("precursorList");
            for (auto precursor : precursorList.children("precursor")) {
                auto selectedIonList = precursor.child("selectedIonList");
                for (auto selectedIon : selectedIonList.children("selectedIon")) {
                    for (auto cvp : selectedIon.children("cvParam")) {
                        std::string acc = cvp.attribute("accession").as_string();
                        if (acc == "MS:1000744") {
                            precursorMz = cvp.attribute("value").as_double();
                        } else if (acc == "MS:1000042") {
                            precursorIntensity = cvp.attribute("value").as_double();
                        }
                    }
                }
            }

            MS2Spectrum sp;
            sp.scanTime          = scanTime;
            sp.mz                = std::move(mzArray);
            sp.intensity         = std::move(intensityArray);
            sp.precursorMz       = precursorMz;
            sp.precursorIntensity = precursorIntensity;
            sp.polarity          = polarity;
            sp.scanId            = scanId;
            result.ms2.push_back(std::move(sp));
        }
    }

    return result;
}
