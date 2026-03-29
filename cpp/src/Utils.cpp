#include "Utils.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <stdexcept>

const std::map<std::string, double> ATOMIC_MASSES = {
    {"H",  1.007825032},
    {"C",  12.0},
    {"N",  14.003074004},
    {"O",  15.994914620},
    {"P",  30.973761998},
    {"S",  31.972071174},
    {"Cl", 34.96885268},
};

namespace Utils {

FormulaTools& getFormulaTools() {
    static FormulaTools ft;
    return ft;
}

std::map<std::string, int> parseMolecularFormula(const std::string& formula) {
    if (formula.empty()) throw std::invalid_argument("Formula must be non-empty");
    return getFormulaTools().parseFormula(formula);
}

double calculateMolecularMass(const std::string& formula) {
    auto composition = parseMolecularFormula(formula);
    return getFormulaTools().calcMolWeight(composition);
}

std::tuple<double, int, int> adductMassChange(const AdductInfo& ai) {
    int charge = ai.charge;
    int multiplier = ai.multiplier < 1 ? 1 : ai.multiplier;

    double massChange = 0.0;
    if (!ai.elementsAdded.empty() || !ai.elementsLost.empty()) {
        double addedMass = ai.elementsAdded.empty() ? 0.0 : calculateMolecularMass(ai.elementsAdded);
        double lostMass  = ai.elementsLost.empty()  ? 0.0 : calculateMolecularMass(ai.elementsLost);
        massChange = addedMass - lostMass - charge * ELECTRON_MASS;
    } else {
        massChange = ai.massChange;
    }

    return {massChange, charge, multiplier};
}

double calculateMzFromFormula(const std::string& formula, const std::string& adduct,
                               const std::vector<AdductInfo>& adductsData) {
    double molecularMass = calculateMolecularMass(formula);

    for (const auto& ai : adductsData) {
        if (ai.adduct == adduct) {
            auto [massChange, charge, multiplier] = adductMassChange(ai);
            return (multiplier * molecularMass + massChange) / std::abs(charge);
        }
    }
    throw std::runtime_error("Unknown adduct: " + adduct);
}

std::pair<double, double> getMassToleranceWindow(double mz, double tolerancePpm) {
    double delta = mz * tolerancePpm / 1e6;
    return {mz - delta, mz + delta};
}

std::vector<std::string> generateColorPalette(int nColors) {
    static const std::vector<std::string> baseColors = {
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d3", "#c7c7c7", "#dbdb8d", "#9edae5",
    };

    std::vector<std::string> colors = baseColors;

    // Generate additional colors using golden ratio if needed
    for (int i = (int)baseColors.size(); i < nColors; ++i) {
        double hue = std::fmod(i * 0.618033988749895, 1.0);
        // Convert HSV to RGB (s=0.7, v=0.9)
        double s = 0.7, v = 0.9;
        int h_i = (int)(hue * 6);
        double f = hue * 6 - h_i;
        double p = v * (1 - s);
        double q = v * (1 - f * s);
        double t = v * (1 - (1 - f) * s);

        double r, g, b;
        switch (h_i % 6) {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            default: r = v; g = p; b = q; break;
        }

        char buf[8];
        std::snprintf(buf, sizeof(buf), "#%02x%02x%02x",
                      (int)(r * 255), (int)(g * 255), (int)(b * 255));
        colors.push_back(buf);
    }

    colors.resize(nColors);
    return colors;
}

std::string formatRetentionTime(double rtMinutes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << rtMinutes << " min";
    return oss.str();
}

std::string formatMz(double mz, int decimals) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(decimals) << mz;
    return oss.str();
}

double calculateCosineSimilarity(const SpectrumData& s1, const SpectrumData& s2,
                                  double mzTolerance) {
    if (s1.mz.empty() || s2.mz.empty()) return 0.0;

    // Build matched peak pairs using Hungarian-like greedy approach
    // For each peak in s1 find the closest unmatched peak in s2 within tolerance
    std::vector<bool> used2(s2.mz.size(), false);

    struct Match {
        double int1, int2;
    };
    std::vector<Match> matches;

    // Collect all candidate pairs sorted by distance
    struct Candidate {
        int i, j;
        double dist;
    };
    std::vector<Candidate> candidates;
    for (int i = 0; i < (int)s1.mz.size(); ++i) {
        for (int j = 0; j < (int)s2.mz.size(); ++j) {
            double dist = std::abs(s1.mz[i] - s2.mz[j]);
            if (dist <= mzTolerance) {
                candidates.push_back({i, j, dist});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });

    std::vector<bool> used1(s1.mz.size(), false);
    for (auto& c : candidates) {
        if (!used1[c.i] && !used2[c.j]) {
            matches.push_back({s1.intensity[c.i], s2.intensity[c.j]});
            used1[c.i] = true;
            used2[c.j] = true;
        }
    }

    if (matches.empty()) return 0.0;

    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (auto& m : matches) {
        dot += m.int1 * m.int2;
    }
    for (double v : s1.intensity) norm1 += v * v;
    for (double v : s2.intensity) norm2 += v * v;

    if (norm1 <= 0.0 || norm2 <= 0.0) return 0.0;
    return dot / (std::sqrt(norm1) * std::sqrt(norm2));
}

SimilarityStats calculateSimilarityStatistics(const std::vector<double>& similarities) {
    if (similarities.empty()) return {0, 0, 0, 0, 0};

    std::vector<double> sorted = similarities;
    std::sort(sorted.begin(), sorted.end());
    int n = (int)sorted.size();

    auto percentile = [&](double p) -> double {
        double idx = p / 100.0 * (n - 1);
        int lo = (int)idx;
        int hi = lo + 1;
        if (hi >= n) return sorted[n - 1];
        double frac = idx - lo;
        return sorted[lo] * (1 - frac) + sorted[hi] * frac;
    };

    return {
        sorted.front(),
        percentile(10),
        percentile(50),
        percentile(90),
        sorted.back()
    };
}

} // namespace Utils
