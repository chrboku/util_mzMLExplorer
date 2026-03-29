#pragma once
#include <string>
#include <vector>
#include <map>
#include <optional>
#include "Utils.h"

struct CompoundEntry {
    std::string name;
    std::string chemicalFormula;
    double mass = 0.0;
    double rtMin = 50.0;         // center RT
    double rtStartMin = 0.0;
    double rtEndMin = 100.0;
    std::string group;
    std::string smiles;
    std::vector<std::string> commonAdducts;
    std::string compoundType; // "formula", "mass", "mz_only"
    std::string isotopologs;
};

struct PrecalculatedAdductData {
    double mz = 0.0;
    std::string polarity;
    std::string displayName;
    bool valid = false;
};

class CompoundManager {
public:
    CompoundManager();

    // Load compounds from a structured data source
    // headers: column names, rows: data rows
    void loadCompounds(const std::vector<std::string>& headers,
                       const std::vector<std::vector<std::string>>& rows);

    // Load adducts from structured data
    void loadAdducts(const std::vector<std::string>& headers,
                     const std::vector<std::vector<std::string>>& rows);

    // Clear all data
    void clearCompounds();

    // Get all compounds
    const std::vector<CompoundEntry>& getCompounds() const { return compounds; }

    // Get compound by name
    const CompoundEntry* getCompoundByName(const std::string& name) const;

    // Get all adducts
    const std::vector<AdductInfo>& getAdducts() const { return adducts; }

    // Get adducts for a compound
    std::vector<std::string> getCompoundAdducts(const std::string& compoundName) const;

    // Calculate m/z for compound-adduct combination
    std::optional<double> calculateCompoundMz(const std::string& compoundName,
                                               const std::string& adduct) const;

    // Get precalculated adduct data
    const PrecalculatedAdductData* getPrecalculatedData(const std::string& compoundName,
                                                         const std::string& adduct) const;

    // Get all available adducts from table
    std::vector<std::string> getAllAvailableAdducts() const;

    // Get RT window for compound
    std::optional<std::tuple<double, double, double>> getCompoundRtWindow(
        const std::string& compoundName) const;

    // Export compounds with m/z to TSV
    void exportCompoundsWithMz(const std::string& outputPath) const;

    // Validate all data
    std::vector<std::string> validateCompoundData() const;

    // Check if a string is an m/z adduct like [197.234]+
    static bool isMzAdduct(const std::string& adductStr);

    // Parse m/z from adduct string
    static std::pair<double, std::string> parseMzAdduct(const std::string& adductStr);

    // Determine polarity from adduct string
    static std::string determinePolarity(const std::string& adduct, int charge = 0);

    bool isEmpty() const { return compounds.empty(); }

private:
    std::vector<CompoundEntry> compounds;
    std::vector<AdductInfo> adducts;

    // Precalculated data: compoundName -> adduct -> data
    std::map<std::string, std::map<std::string, PrecalculatedAdductData>> precalcData;

    void precalculateMzValues();
    std::vector<AdductInfo> getDefaultAdducts() const;

    // Parse adducts string (comma or semicolon separated)
    static std::vector<std::string> parseAdductsString(const std::string& val);

    // Calculate m/z from molecular mass and adduct
    std::optional<double> calculateMzFromMass(double molecularMass,
                                               const std::string& adduct) const;
};
