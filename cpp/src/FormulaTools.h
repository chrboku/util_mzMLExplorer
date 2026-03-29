#pragma once
#include <string>
#include <map>
#include <vector>
#include <stdexcept>

/**
 * FormulaTools - Chemical formula parser and molecular weight calculator.
 * C++ port of the Python FormulaTools class.
 *
 * Supports standard chemical formulas, parentheses, and isotope notation
 * like [13C]C5H12O6.
 */
class FormulaTools {
public:
    struct ElementInfo {
        std::string name;
        std::string symbol;
        int neutrons;
        double mass;
        double abundance;
    };

    FormulaTools();

    // Parse a molecular formula string into an element count map
    // e.g. "C6H12O6" -> {{"C",6},{"H",12},{"O",6}}
    std::map<std::string, int> parseFormula(const std::string& formula) const;

    // Calculate monoisotopic molecular weight from element composition
    double calcMolWeight(const std::map<std::string, int>& composition) const;

    // Convenience: parse formula and calculate weight
    double calcMolWeightFromFormula(const std::string& formula) const;

    // Get all element details
    const std::map<std::string, ElementInfo>& getElementDetails() const { return elemDetails; }

private:
    std::map<std::string, ElementInfo> elemDetails;

    // Internal parsing helpers
    std::pair<int, int> parseNumber(const std::string& formula, int pos) const;
    std::pair<int, std::map<std::string, int>> parseStruct(const std::string& formula, int pos) const;
};
