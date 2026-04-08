#include "FormulaTools.h"
#include <cctype>
#include <stdexcept>
#include <sstream>

FormulaTools::FormulaTools() {
    // Initialize element database
    // Format: symbol -> {name, symbol, neutrons, mass, abundance}
    auto add = [this](const std::string& sym, const std::string& name,
                       int neutrons, double mass, double abundance) {
        elemDetails[sym] = {name, sym, neutrons, mass, abundance};
    };

    add("Al",  "Aluminum",      27,  26.981541,    1.00);
    add("Sb",  "Antimony",      121, 120.903824,   0.573);
    add("Ar",  "Argon",         40,  39.962383,    0.996);
    add("As",  "Arsenic",       75,  74.921596,    1.00);
    add("Ba",  "Barium",        138, 137.905236,   0.717);
    add("Be",  "Beryllium",     9,   9.012183,     1.00);
    add("Bi",  "Bismuth",       209, 208.980388,   1.00);
    add("B",   "Boron",         11,  11.009305,    0.802);
    add("Br",  "Bromine",       79,  78.918336,    0.5069);
    add("Cd",  "Cadmium",       114, 113.903361,   0.2873);
    add("Ca",  "Calcium",       40,  39.962591,    0.9695);
    add("44Ca","Calcium",       44,  43.955485,    0.0208);
    add("C",   "Carbon",        12,  12.0,         0.9893);
    add("12C", "Carbon",        12,  12.0,         0.9893);
    add("13C", "Carbon",        13,  13.00335483507, 0.0107);
    add("Ce",  "Cerium",        140, 139.905442,   0.8848);
    add("Cs",  "Cesium",        133, 132.905433,   1.00);
    add("Cl",  "Chlorine",      35,  34.968853,    0.7577);
    add("35Cl","Chlorine",      35,  34.968853,    0.7577);
    add("37Cl","Chlorine",      37,  36.965903,    0.2423);
    add("Cr",  "Chromium",      52,  51.94051,     0.8379);
    add("50Cr","Chromium",      50,  49.946046,    0.0435);
    add("53Cr","Chromium",      53,  52.940651,    0.095);
    add("54Cr","Chromium",      54,  53.938882,    0.0236);
    add("Co",  "Cobalt",        59,  58.933198,    1.00);
    add("Cu",  "Copper",        63,  62.929599,    0.6917);
    add("65Cu","Copper",        65,  64.927792,    0.3083);
    add("Dy",  "Dysprosium",    164, 163.929183,   0.282);
    add("Er",  "Erbium",        166, 165.930305,   0.336);
    add("Eu",  "Europium",      153, 152.921243,   0.522);
    add("F",   "Fluorine",      19,  18.998403,    1.00);
    add("Gd",  "Gadolinium",    158, 157.924111,   0.2484);
    add("Ga",  "Gallium",       69,  68.925581,    0.601);
    add("Ge",  "Germanium",     74,  73.921179,    0.365);
    add("Au",  "Gold",          197, 196.96656,    1.00);
    add("Hf",  "Hafnium",       180, 179.946561,   0.352);
    add("He",  "Helium",        4,   4.002603,     1.00);
    add("Ho",  "Holmium",       165, 164.930332,   1.00);
    add("H",   "Hydrogen",      1,   1.007825,     0.999);
    add("1H",  "Hydrogen",      1,   1.007825,     0.999);
    add("D",   "Hydrogen",      2,   2.01410177812, 0.001);
    add("2H",  "Hydrogen",      2,   2.01410177812, 0.001);
    add("In",  "Indium",        115, 114.903875,   0.957);
    add("I",   "Iodine",        127, 126.904477,   1.00);
    add("Ir",  "Iridium",       193, 192.962942,   0.627);
    add("Fe",  "Iron",          56,  55.934939,    0.9172);
    add("56Fe","Iron",          56,  55.934939,    0.9172);
    add("54Fe","Iron",          54,  53.939612,    0.058);
    add("57Fe","Iron",          57,  56.935396,    0.022);
    add("Kr",  "Krypton",       84,  83.911506,    0.57);
    add("La",  "Lanthanum",     139, 138.906355,   0.9991);
    add("Pb",  "Lead",          208, 207.976641,   0.524);
    add("Li",  "Lithium",       7,   7.016005,     0.9258);
    add("Lu",  "Lutetium",      175, 174.940785,   0.974);
    add("Mg",  "Magnesium",     24,  23.985045,    0.789);
    add("25Mg","Magnesium",     25,  24.985839,    0.10);
    add("26Mg","Magnesium",     26,  25.982595,    0.111);
    add("Mn",  "Manganese",     55,  54.938046,    1.00);
    add("Hg",  "Mercury",       202, 201.970632,   0.2965);
    add("Mo",  "Molybdenum",    98,  97.905405,    0.2413);
    add("Nd",  "Neodymium",     142, 141.907731,   0.2713);
    add("Ne",  "Neon",          20,  19.992439,    0.906);
    add("Ni",  "Nickel",        58,  57.935347,    0.6827);
    add("Nb",  "Niobium",       93,  92.906378,    1.00);
    add("N",   "Nitrogen",      14,  14.003074,    0.9963);
    add("14N", "Nitrogen",      14,  14.003074,    0.9963);
    add("15N", "Nitrogen",      15,  15.0001088982, 0.00364);
    add("Os",  "Osmium",        192, 191.961487,   0.41);
    add("O",   "Oxygen",        16,  15.994915,    0.9976);
    add("Pd",  "Palladium",     106, 105.903475,   0.2733);
    add("P",   "Phosphorus",    31,  30.973763,    1.00);
    add("Pt",  "Platinum",      195, 194.964785,   0.338);
    add("K",   "Potassium",     39,  38.963708,    0.932);
    add("41K", "Potassium",     41,  40.961825,    0.0673);
    add("Pr",  "Praseodymium",  141, 140.907657,   1.00);
    add("Re",  "Rhenium",       187, 186.955765,   0.626);
    add("Rh",  "Rhodium",       103, 102.905503,   1.00);
    add("Rb",  "Rubidium",      85,  84.9118,      0.7217);
    add("Ru",  "Ruthenium",     102, 101.904348,   0.316);
    add("Sm",  "Samarium",      152, 151.919741,   0.267);
    add("Sc",  "Scandium",      45,  44.955914,    1.00);
    add("Se",  "Selenium",      80,  79.916521,    0.496);
    add("Si",  "Silicon",       28,  27.976928,    0.9223);
    add("Ag",  "Silver",        107, 106.905095,   0.5184);
    add("Na",  "Sodium",        23,  22.98977,     1.00);
    add("Sr",  "Strontium",     88,  87.905625,    0.8258);
    add("S",   "Sulfur",        32,  31.972072,    0.9502);
    add("34S", "Sulfur",        34,  33.967868,    0.0421);
    add("Ta",  "Tantalum",      181, 180.948014,   0.9999);
    add("Te",  "Tellurium",     130, 129.906229,   0.338);
    add("Tb",  "Terbium",       159, 158.92535,    1.00);
    add("Tl",  "Thallium",      205, 204.97441,    0.7048);
    add("Th",  "Thorium",       232, 232.038054,   1.00);
    add("Tm",  "Thulium",       169, 168.934225,   1.00);
    add("Sn",  "Tin",           120, 119.902199,   0.324);
    add("Ti",  "Titanium",      48,  47.947947,    0.738);
    add("W",   "Tungsten",      184, 183.950953,   0.3067);
    add("U",   "Uranium",       238, 238.050786,   0.9927);
    add("V",   "Vanadium",      51,  50.943963,    0.9975);
    add("Xe",  "Xenon",         132, 131.904148,   0.269);
    add("Yb",  "Ytterbium",     174, 173.938873,   0.318);
    add("Y",   "Yttrium",       89,  88.905856,    1.00);
    add("Zn",  "Zinc",          64,  63.929145,    0.486);
    add("66Zn","Zinc",          66,  65.926035,    0.279);
    add("67Zn","Zinc",          67,  66.927129,    0.041);
    add("68Zn","Zinc",          68,  67.924846,    0.188);
    add("Zr",  "Zirconium",     90,  89.904708,    0.5145);
}

std::pair<int, int> FormulaTools::parseNumber(const std::string& formula, int pos) const {
    if (pos >= (int)formula.size()) return {pos, 1};
    if (!std::isdigit(formula[pos])) return {pos, 1};

    std::string numStr;
    while (pos < (int)formula.size() && std::isdigit(formula[pos])) {
        numStr += formula[pos++];
    }
    return {pos, std::stoi(numStr)};
}

std::pair<int, std::map<std::string, int>>
FormulaTools::parseStruct(const std::string& formula, int pos) const {
    std::map<std::string, int> elemDict;

    if (formula[pos] == '(') {
        pos++;
        while (formula[pos] != ')') {
            auto [newPos, elem] = parseStruct(formula, pos);
            pos = newPos;
            for (auto& [k, v] : elem) {
                elemDict[k] += v;
            }
        }
        auto [newPos2, numb] = parseNumber(formula, pos + 1);
        pos = newPos2;
        for (auto& [k, v] : elemDict) v *= numb;
        return {pos, elemDict};

    } else if (formula[pos] == '[') {
        // Isotope notation: [13C]
        pos++;
        std::string numStr;
        while (pos < (int)formula.size() && std::isdigit(formula[pos])) {
            numStr += formula[pos++];
        }
        if (numStr.empty()) throw std::runtime_error("Isotope description wrong");

        std::string curElem;
        curElem += formula[pos];
        if (pos + 1 < (int)formula.size() && std::isalpha(formula[pos + 1]) && std::islower(formula[pos + 1])) {
            curElem += formula[pos + 1];
            pos += 2;
        } else {
            pos++;
        }

        if (formula[pos] != ']') {
            throw std::runtime_error("Malformed isotope: " + formula);
        }
        pos++;

        auto [newPos, numb] = parseNumber(formula, pos);
        pos = newPos;

        std::string isotopeKey = numStr + curElem;
        elemDict[curElem] = numb;
        elemDict[isotopeKey] = numb;
        return {pos, elemDict};

    } else {
        // Regular element
        std::string curElem;
        curElem += formula[pos];
        if (pos + 1 < (int)formula.size() && std::isalpha(formula[pos + 1]) && std::islower(formula[pos + 1])) {
            curElem += formula[pos + 1];
        }
        if (!curElem.empty()) {
            auto [newPos, numb] = parseNumber(formula, pos + (int)curElem.size());
            elemDict[curElem] = numb;
            return {newPos, elemDict};
        }
        throw std::runtime_error("Unrecognized element in formula: " + formula);
    }
}

std::map<std::string, int> FormulaTools::parseFormula(const std::string& formula) const {
    if (formula.empty()) throw std::invalid_argument("Formula must be a non-empty string");
    // Wrap in parentheses like the Python version does
    std::string wrapped = "(" + formula + ")";
    // Remove spaces
    std::string clean;
    for (char c : wrapped) {
        if (c != ' ') clean += c;
    }
    auto [pos, result] = parseStruct(clean, 0);
    return result;
}

double FormulaTools::calcMolWeight(const std::map<std::string, int>& composition) const {
    double mass = 0.0;
    for (auto& [elem, count] : composition) {
        auto it = elemDetails.find(elem);
        if (it != elemDetails.end()) {
            mass += it->second.mass * count;
        }
        // Unknown elements contribute 0 mass
    }
    return mass;
}

double FormulaTools::calcMolWeightFromFormula(const std::string& formula) const {
    return calcMolWeight(parseFormula(formula));
}
