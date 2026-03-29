#include "CompoundManager.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <regex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <set>

CompoundManager::CompoundManager() {
    adducts = getDefaultAdducts();
}

std::vector<AdductInfo> CompoundManager::getDefaultAdducts() const {
    return {
        {"[M+H]+",   0.0, 1, 1, "H",  ""},
        {"[M-H]-",   0.0, -1, 1, "", "H"},
        {"[M+Na]+",  0.0, 1, 1, "Na", ""},
        {"[M+K]+",   0.0, 1, 1, "K",  ""},
        {"[M+NH4]+", 0.0, 1, 1, "NH4",""},
        {"[M-H2O+H]+", 0.0, 1, 1, "H", "H2O"},
        {"[M+2H]2+", 0.0, 2, 1, "H2", ""},
        {"[M-2H]2-", 0.0, -2, 1, "", "H2"},
        {"[M+HCOO]-", 0.0, -1, 1, "HCOO",""},
        {"[M+CH3COO]-", 0.0, -1, 1, "CH3COO",""},
    };
}

void CompoundManager::clearCompounds() {
    compounds.clear();
    precalcData.clear();
}

std::vector<std::string> CompoundManager::parseAdductsString(const std::string& val) {
    if (val.empty() || val == "nan") return {};
    char sep = (val.find(',') != std::string::npos) ? ',' : ';';
    std::vector<std::string> result;
    std::istringstream ss(val);
    std::string tok;
    while (std::getline(ss, tok, sep)) {
        while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) tok.erase(0,1);
        while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t')) tok.pop_back();
        if (!tok.empty()) result.push_back(tok);
    }
    return result;
}

bool CompoundManager::isMzAdduct(const std::string& adductStr) {
    static const std::regex pattern(R"(\[(\d+\.?\d*)\]([+-]))");
    return std::regex_match(adductStr, pattern);
}

std::pair<double, std::string> CompoundManager::parseMzAdduct(const std::string& adductStr) {
    static const std::regex pattern(R"(\[(\d+\.?\d*)\]([+-]))");
    std::smatch m;
    if (std::regex_match(adductStr, m, pattern)) {
        double mz = std::stod(m[1]);
        std::string pol = (m[2] == "+") ? "positive" : "negative";
        return {mz, pol};
    }
    throw std::runtime_error("Not an m/z adduct: " + adductStr);
}

std::string CompoundManager::determinePolarity(const std::string& adduct, int charge) {
    if (!adduct.empty() && adduct.back() == '+') return "positive";
    if (!adduct.empty() && adduct.back() == '-') return "negative";
    if (charge > 0) return "positive";
    if (charge < 0) return "negative";
    return "";
}

void CompoundManager::loadAdducts(const std::vector<std::string>& headers,
                                   const std::vector<std::vector<std::string>>& rows) {
    auto colIdx = [&](const std::string& name) -> int {
        for (int i = 0; i < (int)headers.size(); ++i) {
            std::string h = headers[i];
            std::transform(h.begin(), h.end(), h.begin(), ::tolower);
            if (h == name) return i;
        }
        return -1;
    };

    int adductCol  = colIdx("adduct");
    int chargeCol  = colIdx("charge");
    int multCol    = colIdx("multiplier");
    int mcCol      = colIdx("mass_change");
    int eaCol      = colIdx("elementsadded");
    int elCol      = colIdx("elementslost");

    adducts.clear();
    for (const auto& row : rows) {
        if (adductCol < 0 || adductCol >= (int)row.size()) continue;
        AdductInfo ai;
        ai.adduct = row[adductCol];
        ai.charge = (chargeCol >= 0 && chargeCol < (int)row.size()) ? std::stoi(row[chargeCol]) : 1;
        ai.multiplier = (multCol >= 0 && multCol < (int)row.size()) ? std::stoi(row[multCol]) : 1;
        ai.massChange = (mcCol >= 0 && mcCol < (int)row.size() && !row[mcCol].empty()) ?
                        std::stod(row[mcCol]) : 0.0;
        ai.elementsAdded = (eaCol >= 0 && eaCol < (int)row.size()) ? row[eaCol] : "";
        ai.elementsLost  = (elCol >= 0 && elCol < (int)row.size()) ? row[elCol] : "";
        adducts.push_back(ai);
    }
}

void CompoundManager::loadCompounds(const std::vector<std::string>& headers,
                                     const std::vector<std::vector<std::string>>& rows) {
    auto colIdx = [&](const std::string& name) -> int {
        for (int i = 0; i < (int)headers.size(); ++i) {
            std::string h = headers[i];
            std::transform(h.begin(), h.end(), h.begin(), ::tolower);
            if (h == name) return i;
        }
        return -1;
    };

    int nameCol      = colIdx("name");
    int formulaCol   = colIdx("chemicalformula");
    int massCol      = colIdx("mass");
    int rtMinCol     = colIdx("rt_min");
    int rtStartCol   = colIdx("rt_start_min");
    int rtEndCol     = colIdx("rt_end_min");
    int groupCol     = colIdx("group");
    int smilesCol    = colIdx("smiles");
    int adductsCol   = colIdx("common_adducts");
    int isoCol       = colIdx("isotopologs");

    if (nameCol < 0) throw std::runtime_error("Missing required column: Name");

    std::set<std::string> existingNames;
    for (const auto& c : compounds) existingNames.insert(c.name);

    int addedCount = 0;
    for (const auto& row : rows) {
        if (nameCol >= (int)row.size()) continue;
        std::string name = row[nameCol];
        if (name.empty()) continue;
        if (existingNames.count(name)) {
            std::cout << "Compound already exists, skipping: " << name << "\n";
            continue;
        }

        CompoundEntry e;
        e.name = name;

        auto getField = [&](int idx) -> std::string {
            if (idx >= 0 && idx < (int)row.size()) return row[idx];
            return "";
        };

        std::string formula = getField(formulaCol);
        std::string massStr = getField(massCol);

        e.chemicalFormula = formula;
        if (!formula.empty()) {
            try {
                Utils::parseMolecularFormula(formula);
                e.compoundType = "formula";
            } catch (...) {
                std::cout << "Warning: Invalid formula for " << name << ": " << formula << "\n";
                e.compoundType = "formula";
            }
        } else if (!massStr.empty()) {
            try {
                e.mass = std::stod(massStr);
                e.compoundType = "mass";
            } catch (...) {
                std::cout << "Warning: Invalid mass for " << name << ": " << massStr << "\n";
                continue;
            }
        } else {
            // Check for m/z adducts
            std::string adductStr = getField(adductsCol);
            auto adductList = parseAdductsString(adductStr);
            bool hasMzAdducts = false;
            for (const auto& a : adductList) {
                if (isMzAdduct(a)) { hasMzAdducts = true; break; }
            }
            if (hasMzAdducts) {
                e.compoundType = "mz_only";
            } else {
                std::cout << "Warning: No formula, mass, or m/z adducts for " << name << "\n";
                continue;
            }
        }

        // Parse RT values
        auto parseDouble = [](const std::string& s, double def) -> double {
            if (s.empty()) return def;
            try { return std::stod(s); } catch (...) { return def; }
        };

        e.rtMin      = parseDouble(getField(rtMinCol), 50.0);
        e.rtStartMin = parseDouble(getField(rtStartCol), 0.0);
        e.rtEndMin   = parseDouble(getField(rtEndCol), 100.0);
        e.group      = getField(groupCol);
        e.smiles     = getField(smilesCol);
        e.isotopologs = getField(isoCol);
        e.commonAdducts = parseAdductsString(getField(adductsCol));

        compounds.push_back(e);
        existingNames.insert(name);
        addedCount++;
    }

    std::cout << "Added " << addedCount << " compounds. Total: " << compounds.size() << "\n";
    precalculateMzValues();
}

const CompoundEntry* CompoundManager::getCompoundByName(const std::string& name) const {
    for (const auto& c : compounds) {
        if (c.name == name) return &c;
    }
    return nullptr;
}

std::optional<double> CompoundManager::calculateMzFromMass(double molMass,
                                                             const std::string& adduct) const {
    for (const auto& ai : adducts) {
        if (ai.adduct == adduct) {
            auto [massChange, charge, multiplier] = Utils::adductMassChange(ai);
            if (charge == 0) return std::nullopt;
            return (multiplier * molMass + massChange) / std::abs(charge);
        }
    }
    return std::nullopt;
}

std::optional<double> CompoundManager::calculateCompoundMz(const std::string& compoundName,
                                                             const std::string& adduct) const {
    const CompoundEntry* c = getCompoundByName(compoundName);
    if (!c) return std::nullopt;

    if (isMzAdduct(adduct)) {
        auto [mz, pol] = parseMzAdduct(adduct);
        return mz;
    }

    double molMass = 0.0;
    if (c->compoundType == "formula") {
        try {
            molMass = Utils::calculateMolecularMass(c->chemicalFormula);
        } catch (...) { return std::nullopt; }
    } else if (c->compoundType == "mass") {
        molMass = c->mass;
    } else {
        return std::nullopt;
    }

    return calculateMzFromMass(molMass, adduct);
}

void CompoundManager::precalculateMzValues() {
    precalcData.clear();
    auto allAdducts = getAllAvailableAdducts();

    for (const auto& c : compounds) {
        precalcData[c.name] = {};

        std::vector<std::string> toCalc;
        if (c.compoundType == "mz_only") {
            toCalc = c.commonAdducts;
        } else {
            toCalc = c.commonAdducts;
            for (const auto& a : allAdducts) {
                if (std::find(toCalc.begin(), toCalc.end(), a) == toCalc.end())
                    toCalc.push_back(a);
            }
        }

        for (const auto& adduct : toCalc) {
            PrecalculatedAdductData pd;
            auto mz = calculateCompoundMz(c.name, adduct);
            if (mz.has_value()) {
                pd.mz = mz.value();
                pd.valid = true;
            }

            if (isMzAdduct(adduct)) {
                auto [mzVal, pol] = parseMzAdduct(adduct);
                pd.polarity = pol;
                pd.displayName = "m/z " + Utils::formatMz(mzVal) + " (" + pol + ")";
            } else {
                pd.polarity = determinePolarity(adduct);
                pd.displayName = adduct;
            }

            precalcData[c.name][adduct] = pd;
        }
    }
}

const PrecalculatedAdductData* CompoundManager::getPrecalculatedData(
    const std::string& compoundName, const std::string& adduct) const {
    auto it = precalcData.find(compoundName);
    if (it == precalcData.end()) return nullptr;
    auto it2 = it->second.find(adduct);
    if (it2 == it->second.end()) return nullptr;
    return &it2->second;
}

std::vector<std::string> CompoundManager::getCompoundAdducts(const std::string& compoundName) const {
    const CompoundEntry* c = getCompoundByName(compoundName);
    if (!c) return {};
    if (!c->commonAdducts.empty()) return c->commonAdducts;
    return getAllAvailableAdducts();
}

std::vector<std::string> CompoundManager::getAllAvailableAdducts() const {
    std::vector<std::string> result;
    for (const auto& a : adducts) result.push_back(a.adduct);
    return result;
}

std::optional<std::tuple<double, double, double>>
CompoundManager::getCompoundRtWindow(const std::string& compoundName) const {
    const CompoundEntry* c = getCompoundByName(compoundName);
    if (!c) return std::nullopt;
    return std::make_tuple(c->rtMin, c->rtStartMin, c->rtEndMin);
}

void CompoundManager::exportCompoundsWithMz(const std::string& outputPath) const {
    std::ofstream ofs(outputPath);
    if (!ofs) throw std::runtime_error("Cannot write to: " + outputPath);

    ofs << "Name\tChemicalFormula\tMass\tAdduct\tMZ\tRT_min\tRT_start_min\tRT_end_min\tGroup\n";
    for (const auto& c : compounds) {
        auto adductList = getCompoundAdducts(c.name);
        for (const auto& adduct : adductList) {
            auto mz = calculateCompoundMz(c.name, adduct);
            ofs << c.name << "\t"
                << c.chemicalFormula << "\t"
                << (c.mass > 0 ? std::to_string(c.mass) : "") << "\t"
                << adduct << "\t"
                << (mz.has_value() ? Utils::formatMz(mz.value()) : "N/A") << "\t"
                << std::fixed << std::setprecision(2)
                << c.rtMin << "\t" << c.rtStartMin << "\t" << c.rtEndMin << "\t"
                << c.group << "\n";
        }
    }
}

std::vector<std::string> CompoundManager::validateCompoundData() const {
    std::vector<std::string> issues;
    for (const auto& c : compounds) {
        if (c.rtStartMin >= c.rtEndMin)
            issues.push_back(c.name + ": RT_start_min must be less than RT_end_min");
        if (c.rtMin < c.rtStartMin || c.rtMin > c.rtEndMin)
            issues.push_back(c.name + ": RT_min should be between RT_start_min and RT_end_min");
    }
    return issues;
}
