#pragma once
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <optional>
#include "FormulaTools.h"

// Monoisotopic electron mass in Da
constexpr double ELECTRON_MASS = 0.000549;

// Atomic masses (monoisotopic)
extern const std::map<std::string, double> ATOMIC_MASSES;

// Adduct information structure
struct AdductInfo {
    std::string adduct;
    double massChange;
    int charge;
    int multiplier;
    std::string elementsAdded;
    std::string elementsLost;
};

// Spectrum data for cosine similarity
struct SpectrumData {
    std::vector<double> mz;
    std::vector<double> intensity;
    double precursorMz = 0.0;
};

// Utility functions
namespace Utils {

// Parse molecular formula
std::map<std::string, int> parseMolecularFormula(const std::string& formula);

// Calculate monoisotopic molecular mass
double calculateMolecularMass(const std::string& formula);

// Calculate adduct mass change, charge, and multiplier from AdductInfo
std::tuple<double, int, int> adductMassChange(const AdductInfo& adductInfo);

// Calculate m/z from formula and adduct
double calculateMzFromFormula(const std::string& formula, const std::string& adduct,
                               const std::vector<AdductInfo>& adductsData);

// Calculate mass tolerance window
std::pair<double, double> getMassToleranceWindow(double mz, double tolerancePpm = 5.0);

// Generate color palette
std::vector<std::string> generateColorPalette(int nColors);

// Format retention time
std::string formatRetentionTime(double rtMinutes);

// Format m/z value
std::string formatMz(double mz, int decimals = 4);

// Calculate cosine similarity between two spectra
double calculateCosineSimilarity(const SpectrumData& spectrum1, const SpectrumData& spectrum2,
                                  double mzTolerance = 0.1);

// Calculate similarity statistics
struct SimilarityStats {
    double min;
    double percentile10;
    double median;
    double percentile90;
    double max;
};
SimilarityStats calculateSimilarityStatistics(const std::vector<double>& similarities);

// Get global FormulaTools instance
FormulaTools& getFormulaTools();

} // namespace Utils
