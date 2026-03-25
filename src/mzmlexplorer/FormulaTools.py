# a class used to parse chemical formulas
# e.g. the formula C6H12O6 will be parsed to a dictionary {'H':12, 'C':6, 'O':6}
# NOTE: different isotopes may be specified as [13C]C5H12O6
# fmt: off
from math import comb


class formulaTools:
    def __init__(self, elemDetails=None):
        if elemDetails is None:
            self.elemDetails = {}
            #              Element      Name        short Neutrons Mass    Abundance
            self.elemDetails["Al"]   = ["Aluminum", "Al", 27, 26.981541, 1.00]
            self.elemDetails["Sb"]   = ["Antimony", "Sb", 121, 120.903824, 0.573]
            self.elemDetails["Ar"]   = ["Argon", "Ar", 40, 39.962383, 0.996]
            self.elemDetails["As"]   = ["Arsenic", "As", 75, 74.921596, 1.00]
            self.elemDetails["Ba"]   = ["Barium", "Ba", 138, 137.905236, 0.717]
            self.elemDetails["Be"]   = ["Beryllium", "Be", 9, 9.012183, 1.00]
            self.elemDetails["Bi"]   = ["Bismuth", "Bi", 209, 208.980388, 1.00]
            self.elemDetails["B"]    = ["Boron", "B", 11, 11.009305, 0.802]
            self.elemDetails["Br"]   = ["Bromine", "Br", 79, 78.918336, 0.5069]
            self.elemDetails["Cd"]   = ["Cadmium", "Cd", 114, 113.903361, 0.2873]
            self.elemDetails["Ca"]   = ["Calcium", "Ca", 40, 39.962591, 0.9695]
            self.elemDetails["44Ca"] = ["Calcium","Ca",44,43.955485,0.0208,]  # 3.992894
            self.elemDetails["C"]    = ["Carbon", "C", 12, 12.0, 0.9893]
            self.elemDetails["12C"]  = ["Carbon", "C", 12, 12.0, 0.9893]
            self.elemDetails["13C"]  = ["Carbon","C",13,13.00335483507,0.0107,]  # 1.00335
            self.elemDetails["Ce"]   = ["Cerium", "Ce", 140, 139.905442, 0.8848]
            self.elemDetails["Cs"]   = ["Cesium", "Cs", 133, 132.905433, 1.00]
            self.elemDetails["Cl"]   = ["Chlorine", "Cl", 35, 34.968853, 0.7577]
            self.elemDetails["35Cl"] = ["Chlorine", "Cl", 35, 34.968853, 0.7577]
            self.elemDetails["37Cl"] = ["Chlorine","Cl",37,36.965903,0.2423,]  # 1.997077
            self.elemDetails["Cr"]   = ["Chromium", "Cr", 52, 51.94051, 0.8379]
            self.elemDetails["50Cr"] = ["Chromium","Cr",50,49.946046,0.0435,]  # -1.994464
            self.elemDetails["53Cr"] = ["Chromium","Cr",53,52.940651,0.095,]  # 1.000141
            self.elemDetails["54Cr"] = ["Chromium","Cr",54,53.938882,0.0236,]  # 1.998372
            self.elemDetails["Co"]   = ["Cobalt", "Co", 59, 58.933198, 1.00]
            self.elemDetails["Cu"]   = ["Copper", "Cu", 63, 62.929599, 0.6917]
            self.elemDetails["65Cu"] = ["Copper","Cu",65,64.927792,0.3083,]  # 1.998193
            self.elemDetails["Dy"]   = ["Dysprosium", "Dy", 164, 163.929183, 0.282]
            self.elemDetails["Er"]   = ["Erbium", "Er", 166, 165.930305, 0.336]
            self.elemDetails["Eu"]   = ["Europium", "Eu", 153, 152.921243, 0.522]
            self.elemDetails["F"]    = ["Fluorine", "F", 19, 18.998403, 1.00]
            self.elemDetails["Gd"]   = ["Gadolinium", "Gd", 158, 157.924111, 0.2484]
            self.elemDetails["Ga"]   = ["Gallium", "Ga", 69, 68.925581, 0.601]
            self.elemDetails["Ge"]   = ["Germanium", "Ge", 74, 73.921179, 0.365]
            self.elemDetails["Au"]   = ["Gold", "Au", 197, 196.96656, 1.00]
            self.elemDetails["Hf"]   = ["Hafnium", "Hf", 180, 179.946561, 0.352]
            self.elemDetails["He"]   = ["Helium", "He", 4, 4.002603, 1.00]
            self.elemDetails["Ho"]   = ["Holmium", "Ho", 165, 164.930332, 1.00]
            self.elemDetails["H"]    = ["Hydrogen", "H", 1, 1.007825, 0.999]
            self.elemDetails["1H"]   = ["Hydrogen", "H", 1, 1.007825, 0.999]
            self.elemDetails["D"]    = ["Hydrogen","H",2,2.01410177812,0.001,]  
            self.elemDetails["2H"]   = ["Hydrogen","H",2,2.01410177812,0.001,]  
            self.elemDetails["In"]   = ["Indium", "In", 115, 114.903875, 0.957]
            self.elemDetails["I"]    = ["Iodine", "I", 127, 126.904477, 1.00]
            self.elemDetails["Ir"]   = ["Iridium", "Ir", 193, 192.962942, 0.627]
            self.elemDetails["Fe"]   = ["Iron", "Fe", 56, 55.934939, 0.9172]
            self.elemDetails["56Fe"] = ["Iron", "Fe", 56, 55.934939, 0.9172]
            self.elemDetails["54Fe"] = ["Iron", "Fe", 54, 53.939612, 0.058]  # -1.995327
            self.elemDetails["57Fe"] = ["Iron", "Fe", 57, 56.935396, 0.022]  # 1.000457
            self.elemDetails["Kr"]   = ["Krypton", "Kr", 84, 83.911506, 0.57]
            self.elemDetails["La"]   = ["Lanthanum", "La", 139, 138.906355, 0.9991]
            self.elemDetails["Pb"]   = ["Lead", "Pb", 208, 207.976641, 0.524]
            self.elemDetails["Li"]   = ["Lithium", "Li", 7, 7.016005, 0.9258]
            self.elemDetails["Lu"]   = ["Lutetium", "Lu", 175, 174.940785, 0.974]
            self.elemDetails["Mg"]   = ["Magnesium", "Mg", 24, 23.985045, 0.789]
            self.elemDetails["25Mg"] = ["Magnesium","Mg",25,24.985839,0.10,]  # 1.000794
            self.elemDetails["26Mg"] = ["Magnesium","Mg",26,25.982595,0.111,]  # 1.99755
            self.elemDetails["Mn"]   = ["Manganese", "Mn", 55, 54.938046, 1.00]
            self.elemDetails["Hg"]   = ["Mercury", "Hg", 202, 201.970632, 0.2965]
            self.elemDetails["Mo"]   = ["Molybdenum", "Mo", 98, 97.905405, 0.2413]
            self.elemDetails["Nd"]   = ["Neodymium", "Nd", 142, 141.907731, 0.2713]
            self.elemDetails["Ne"]   = ["Neon", "Ne", 20, 19.992439, 0.906]
            self.elemDetails["Ni"]   = ["Nickel", "Ni", 58, 57.935347, 0.6827]
            self.elemDetails["Nb"]   = ["Niobium", "Nb", 93, 92.906378, 1.00]
            self.elemDetails["N"]    = ["Nitrogen", "N", 14, 14.003074, 0.9963]
            self.elemDetails["14N"]  = ["Nitrogen", "N", 14, 14.003074, 0.9963]
            self.elemDetails["15N"]  = ["Nitrogen", "N", 15, 15.0001088982, 0.00364]
            self.elemDetails["Os"]   = ["Osmium", "Os", 192, 191.961487, 0.41]
            self.elemDetails["O"]    = ["Oxygen", "O", 16, 15.994915, 0.9976]
            self.elemDetails["Pd"]   = ["Palladium", "Pd", 106, 105.903475, 0.2733]
            self.elemDetails["P"]    = ["Phosphorus", "P", 31, 30.973763, 1.00]
            self.elemDetails["Pt"]   = ["Platinum", "Pt", 195, 194.964785, 0.338]
            self.elemDetails["K"]    = ["Potassium", "K", 39, 38.963708, 0.932]
            self.elemDetails["41K"]  = ["Potassium","K",41,40.961825,0.0673,]  # 1.998117
            self.elemDetails["Pr"]   = ["Praseodymium", "Pr", 141, 140.907657, 1.00]
            self.elemDetails["Re"]   = ["Rhenium", "Re", 187, 186.955765, 0.626]
            self.elemDetails["Rh"]   = ["Rhodium", "Rh", 103, 102.905503, 1.00]
            self.elemDetails["Rb"]   = ["Rubidium", "Rb", 85, 84.9118, 0.7217]
            self.elemDetails["Ru"]   = ["Ruthenium", "Ru", 102, 101.904348, 0.316]
            self.elemDetails["Sm"]   = ["Samarium", "Sm", 152, 151.919741, 0.267]
            self.elemDetails["Sc"]   = ["Scandium", "Sc", 45, 44.955914, 1.00]
            self.elemDetails["Se"]   = ["Selenium", "Se", 80, 79.916521, 0.496]
            self.elemDetails["Si"]   = ["Silicon", "Si", 28, 27.976928, 0.9223]
            self.elemDetails["Ag"]   = ["Silver", "Ag", 107, 106.905095, 0.5184]
            self.elemDetails["Na"]   = ["Sodium", "Na", 23, 22.98977, 1.00]
            self.elemDetails["Sr"]   = ["Strontium", "Sr", 88, 87.905625, 0.8258]
            self.elemDetails["S"]    = ["Sulfur", "S", 32, 31.972072, 0.9502]
            self.elemDetails["34S"]  = ["Sulfur", "S", 34, 33.967868, 0.0421]  # 1.995796
            self.elemDetails["Ta"]   = ["Tantalum", "Ta", 181, 180.948014, 0.9999]
            self.elemDetails["Te"]   = ["Tellurium", "Te", 130, 129.906229, 0.338]
            self.elemDetails["Tb"]   = ["Terbium", "Tb", 159, 158.92535, 1.00]
            self.elemDetails["Tl"]   = ["Thallium", "Tl", 205, 204.97441, 0.7048]
            self.elemDetails["Th"]   = ["Thorium", "Th", 232, 232.038054, 1.00]
            self.elemDetails["Tm"]   = ["Thulium", "Tm", 169, 168.934225, 1.00]
            self.elemDetails["Sn"]   = ["Tin", "Sn", 120, 119.902199, 0.324]
            self.elemDetails["Ti"]   = ["Titanium", "Ti", 48, 47.947947, 0.738]
            self.elemDetails["W"]    = ["Tungsten", "W", 184, 183.950953, 0.3067]
            self.elemDetails["U"]    = ["Uranium", "U", 238, 238.050786, 0.9927]
            self.elemDetails["V"]    = ["Vanadium", "V", 51, 50.943963, 0.9975]
            self.elemDetails["Xe"]   = ["Xenon", "Xe", 132, 131.904148, 0.269]
            self.elemDetails["Yb"]   = ["Ytterbium", "Yb", 174, 173.938873, 0.318]
            self.elemDetails["Y"]    = ["Yttrium", "Y", 89, 88.905856, 1.00]
            self.elemDetails["Zn"]   = ["Zinc", "Zn", 64, 63.929145, 0.486]
            self.elemDetails["66Zn"] = ["Zinc", "Zn", 66, 65.926035, 0.279]  # 1.99689
            self.elemDetails["67Zn"] = ["Zinc", "Zn", 67, 66.927129, 0.041]  # 2.997984
            self.elemDetails["68Zn"] = ["Zinc", "Zn", 68, 67.924846, 0.188]  # 3.995701
            self.elemDetails["Zr"]   = ["Zirconium", "Zr", 90, 89.904708, 0.5145]

        else:
            self.elemDetails = elemDetails
    # fmt: on

    # INTERNAL METHOD used for parsing
    # parses a number
    def _parseNumber(self, formula, pos):
        if pos >= len(formula):
            return -1, 1

        if formula[pos].isdigit():
            num = ""
            while formula[pos].isdigit() and pos < len(formula):
                num = num + formula[pos]
                pos = pos + 1
            return pos, int(num)
        else:
            return pos, 1

    # INTERNAL METHOD used for parsing
    # parses an element
    def _parseStruct(self, formula, pos):
        elemDict = {}

        if formula[pos] == "(":
            pos = pos + 1
            while formula[pos] != ")":
                pos, elem = self._parseStruct(formula, pos)
                for kE in elem.keys():
                    if kE in elemDict.keys():
                        elemDict[kE] = elemDict[kE] + elem[kE]
                    else:
                        elemDict[kE] = elem[kE]
            pos, numb = self._parseNumber(formula, pos + 1)
            for kE in elemDict.keys():
                elemDict[kE] = elemDict[kE] * numb
            return pos, elemDict
        elif formula[pos] == "[":
            pos = pos + 1

            num = ""
            while formula[pos].isdigit() and pos < len(formula):
                num = num + formula[pos]
                pos = pos + 1
            if pos == "":
                raise Exception("Isotope description wrong")

            curElem = formula[pos]
            if (pos + 1) < len(formula) and formula[pos + 1].isalpha() and formula[pos + 1].islower():
                curElem = formula[pos : (pos + 2)]
            if curElem != "":
                pos = pos + len(curElem)
            else:
                raise Exception("Unrecognized element")

            if formula[pos] != "]":
                raise Exception("Malformed isotope: " + formula)
            pos = pos + 1

            pos, numb = self._parseNumber(formula, pos)
            elemDict[curElem] = numb
            elemDict[num + curElem] = numb

            return pos, elemDict

        else:
            curElem = formula[pos]
            if (pos + 1) < len(formula) and formula[pos + 1].isalpha() and formula[pos + 1].islower():
                curElem = formula[pos : (pos + 2)]
            if curElem != "":
                pos, numb = self._parseNumber(formula, pos + len(curElem))
                elemDict[curElem] = numb
                return pos, elemDict
            else:
                raise Exception("Unrecognized element")
        return -1

    # parses a formula into an element-dictionary
    def parseFormula(self, formula):
        return self._parseStruct("(" + formula.replace(" ", "") + ")", 0)[1]

    # method determines if a given element represent an isotope other the the main isotope of a given element
    # e.g. isIso("13C"): True; isIso("12C"): False
    def isIso(self, iso):
        return not (iso[0].isalpha())

    # returns the element for a given isotope
    def getElementFor(self, elem):
        if not (self.isIso(elem)):
            raise Exception("Element was provided. Isotope is required")
        else:
            num = ""
            pos = 0
            while elem[pos].isdigit() and pos < len(elem):
                num = num + elem[pos]
                pos = pos + 1
            if pos == "":
                raise Exception("Isotope description wrong")
            curElem = elem[pos]

            if (pos + 1) < len(elem) and elem[pos + 1].isalpha() and elem[pos + 1].islower():
                curElem = elem[pos : (pos + 2)]
            if curElem != "":
                pos = pos + len(curElem)
            else:
                raise Exception("Unrecognized element")

            return curElem, num

    # helper method: calculates the isotopic ratio
    def getIsotopologueRatio(self, c, s, p):
        return pow(p, s) * comb(c, s)

    def getMassOffset(self, elems):
        fElems = {}
        fIso = {}
        ret = 0
        for elem in elems:
            if not (self.isIso(elem)):
                fElems[elem] = elems[elem]
        for elem in elems:
            if self.isIso(elem):
                curElem, iso = self.getElementFor(elem)

                if not (fIso.has_key(curElem)):
                    fIso[curElem] = []
                fIso[curElem].append((iso, elems[elem]))

        for elem in fElems:
            rem = 0
            if fIso.has_key(elem):
                for x in fIso[elem]:
                    rem = rem + x[1]
            p = self.elemDetails[elem][4]
            c = fElems[elem] - rem
            ret = ret * pow(p, c)
        for iso in fIso:
            for cIso in fIso[iso]:
                ret = ret + (self.elemDetails[str(cIso[0] + iso)][3] - self.elemDetails[iso][3]) * cIso[1]
        return ret

    def getAbundance(self, elems):
        fElems = {}
        fIso = {}
        ret = 1.0
        for elem in elems:
            if not (self.isIso(elem)):
                fElems[elem] = elems[elem]
        for elem in elems:
            if self.isIso(elem):
                curElem, iso = self.getElementFor(elem)

                if not (fIso.has_key(curElem)):
                    fIso[curElem] = []
                fIso[curElem].append((iso, elems[elem]))
        for elem in fElems:
            rem = 0
            if fIso.has_key(elem):
                for x in fIso[elem]:
                    rem = rem + x[1]
            p = self.elemDetails[elem][4]
            c = fElems[elem] - rem
            ret = ret * pow(p, c)
        for iso in fIso:
            for cIso in fIso[iso]:
                ret = ret * self.getIsotopologueRatio(fElems[iso], cIso[1], self.elemDetails[str(cIso[0]) + iso][4])
        return ret

    def getAbundanceToMonoisotopic(self, elems):
        onlyElems = {}
        for elem in elems:
            if not (self.isIso(elem)):
                onlyElems[elem] = elems[elem]

        return self.getAbundance(elems) / self.getAbundance(onlyElems)

    # calculates the molecular weight of a given elemental collection (e.g. result of parseFormula)
    def calcMolWeight(self, elems):
        mw = 0.0
        for elem in elems.keys():
            if not (self.isIso(elem)):
                mw = mw + self.elemDetails[elem][3] * elems[elem]
            else:
                curElem, iso = self.getElementFor(elem)
                mw = mw + self.elemDetails[iso + curElem][3] * elems[elem] - self.elemDetails[curElem][3] * elems[elem]

        return mw

    # returns putaive isotopes for a given mz difference
    def getPutativeIsotopes(self, mzdiff, atMZ, z=1, ppm=5.0, maxIsoCombinations=1, used=[]):
        mzdiff = mzdiff * z
        maxIsoCombinations = maxIsoCombinations - 1

        ret = []

        for elem in self.elemDetails:
            if self.isIso(elem):
                curElem, iso = self.getElementFor(elem)
                diff = self.elemDetails[iso + curElem][3] - self.elemDetails[curElem][3]

                if maxIsoCombinations == 0:
                    if abs(mzdiff - diff) < (atMZ * ppm / 1000000.0):
                        x = [y for y in used]
                        x.append(iso + curElem)

                        d = {}
                        for y in x:
                            if not (d.has_key(y)):
                                d[y] = 0
                            d[y] = d[y] + 1
                        ret.append(d)

                else:
                    # print "next level with", mzdiff-diff, "after", iso+elem, self.elemDetails[iso+elem][3],self.elemDetails[elem][3]
                    x = [y for y in used]
                    x.append(iso + curElem)
                    x = self.getPutativeIsotopes(
                        mzdiff - diff,
                        atMZ=atMZ,
                        z=1,
                        ppm=ppm,
                        maxIsoCombinations=maxIsoCombinations,
                        used=x,
                    )
                    ret.extend(x)

        if maxIsoCombinations > 0:
            x = [y for y in used]
            x = self.getPutativeIsotopes(
                mzdiff,
                atMZ=atMZ,
                z=1,
                ppm=ppm,
                maxIsoCombinations=maxIsoCombinations,
                used=x,
            )
            ret.extend(x)

        return ret

    # prints a given elemental collection in form of a sum formula
    def flatToString(self, elems, prettyPrintWithHTMLTags=False, subStart="<sub>", subEnd="</sub>"):
        if isinstance(elems, str):
            elems = self.parseFormula(elems)

        if not prettyPrintWithHTMLTags:
            subStart = ""
            subEnd = ""

        fElems = {}
        for elem in elems:
            if not (self.isIso(elem)):
                fElems[elem] = elems[elem]
        for elem in elems:
            if self.isIso(elem):
                curElem, iso = self.getElementFor(elem)

                if fElems.has_key(curElem):
                    fElems[curElem] = fElems[curElem] - elems[elem]
                    fElems["[" + iso + curElem + "]"] = elems[elem]
                else:
                    fElems["[" + iso + curElem + "]"] = elems[elem]

        return "".join([("%s%s%d%s" % (e, subStart, fElems[e], subEnd) if fElems[e] > 1 else "%s" % e) for e in sorted(fElems.keys())])

    def getIsotopes(self, minInt=0.02):
        ret = {}
        for elem in self.elemDetails:
            if self.isIso(elem):
                el = self.getElementFor(elem)
                prob = self.elemDetails[elem][4] / self.elemDetails[el[0]][4]
                if prob >= minInt:
                    ret[elem] = (
                        self.elemDetails[elem][3] - self.elemDetails[el[0]][3],
                        prob,
                    )

        return ret

    def calcDifferenceBetweenElemDicts(self, elemsFragment, elemsParent):
        loss = {}
        for elem in elemsParent:
            l = 0
            if elem in elemsFragment:
                l = elemsParent[elem] - elemsFragment[elem]
            else:
                l = elemsParent[elem]
            if l > 0:
                loss[elem] = l
        return loss

    def calcDifferenceBetweenSumFormulas(self, sfFragment, sfParent):
        return self.calcDifferenceBetweenElemDicts(self.parseFormula(sfFragment), self.parseFormula(sfParent))


class FragmentAnnotator:
    """Annotates MS/MS fragment ions by finding possible sum formulas.

    For each observed fragment m/z, all elemental formulas are enumerated
    whose neutral monoisotopic mass matches within a given ppm tolerance and
    whose element counts do not exceed those of the intact precursor molecule.
    Neutral losses relative to the precursor neutral mass are computed
    analogously.

    Parameters
    ----------
    ft : formulaTools, optional
        Shared formulaTools instance. A new one is created when omitted.

    Example
    -------
    >>> fa = FragmentAnnotator()
    >>> results = fa.annotate("C9H11NO2", [120.0813, 77.0391], ppm=5.0)
    >>> # results[0]["fragment_formulas"] -> list of (formula_str, mass, ppm_err)
    """

    PROTON_MASS = 1.007276  # Da, monoisotopic proton

    def __init__(self, ft=None):
        self.ft = ft if ft is not None else formulaTools()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(
        self,
        precursor_formula,
        fragment_mzs,
        ppm=5.0,
        charge=1,
        ion_mode="positive",
        max_results=10,
        adduct_info=None,
        extra_formula=None,
        tol_da=None,
    ):
        """Annotate a list of fragment m/z values.

        Parameters
        ----------
        precursor_formula : str
            Molecular formula of the intact (neutral) precursor molecule
            (e.g. ``"C9H11NO2"``).  This defines the base upper bound on each
            element count for neutral-loss formula candidates.
        fragment_mzs : iterable of float
            Observed fragment m/z values to annotate.
        ppm : float
            Mass tolerance in parts per million.
        charge : int
            Fragment ion charge state (default 1).
        ion_mode : str
            ``"positive"`` (default) or ``"negative"``.
        max_results : int
            Maximum number of candidate formulas returned per fragment.
        adduct_info : dict or pandas.Series, optional
            Adduct descriptor containing at minimum ``"ElementsAdded"`` and
            ``"ElementsLost"`` (formula strings).  When provided, the element
            upper bounds for *fragment* formula candidates are extended by the
            adduct's added elements and reduced by its lost elements, reflecting
            the full ion composition.  Neutral-loss upper bounds continue to
            use only the neutral molecule's element counts.

        Returns
        -------
        list of dict, one entry per input m/z, each containing:

        ``mz``
            The input fragment m/z value.
        ``fragment_formulas``
            List of ``(formula_str, neutral_mass, ppm_err)`` tuples for
            candidate fragment formulas, sorted by ``|ppm_err|``.
        ``neutral_loss_formulas``
            Same format, for the matched neutral-loss formulas.
            Upper-bounded by the neutral molecule's element counts.
        """
        precursor_elems = self.ft.parseFormula(precursor_formula)
        # Base element counts from the neutral molecule (used for neutral-loss bound)
        mol_elems = {k: v for k, v in precursor_elems.items() if not self.ft.isIso(k)}
        precursor_neutral_mass = self.ft.calcMolWeight(precursor_elems)

        ion_offset = self.PROTON_MASS * charge if ion_mode == "positive" else -self.PROTON_MASS * charge

        # Build ion element upper bound (fragment formulas include adduct atoms)
        ion_elems = dict(mol_elems)
        if adduct_info is not None:
            ion_elems = self._apply_adduct_elements(ion_elems, adduct_info)

        # Expand upper bounds with user-supplied extra elements
        if extra_formula:
            extra = {k: v for k, v in self.ft.parseFormula(extra_formula).items() if not self.ft.isIso(k)}
            for elem, cnt in extra.items():
                mol_elems[elem] = mol_elems.get(elem, 0) + cnt
                ion_elems[elem] = ion_elems.get(elem, 0) + cnt

        def _build(elems):
            order = sorted(
                elems.keys(),
                key=lambda e: self.ft.elemDetails[e][3],
                reverse=True,
            )
            rmf = [0.0] * (len(order) + 1)
            for i in range(len(order) - 1, -1, -1):
                rmf[i] = rmf[i + 1] + elems[order[i]] * self.ft.elemDetails[order[i]][3]
            return order, rmf

        ion_order, ion_rmf = _build(ion_elems)
        mol_order, mol_rmf = _build(mol_elems)

        results = []
        # When tol_da was derived from precursor_mz (tol_da = precursor_mz * ppm / 1e6),
        # we recover that reference mass so that displayed ppm errors are consistent
        # with the threshold (i.e. always ≤ ppm_threshold).
        ref_for_ppm = (tol_da / ppm * 1e6) if (tol_da is not None and ppm > 0) else None

        for mz in fragment_mzs:
            mz_f = float(mz)
            frag_neutral = mz_f * charge - ion_offset

            # Use caller-supplied absolute Da tolerance (e.g. derived from precursor m/z),
            # falling back to computing it per-fragment from ppm when not provided.
            search_tol_da = tol_da if tol_da is not None else frag_neutral * ppm / 1e6

            frag_formulas = self._search(
                frag_neutral, ion_elems, ion_order, ion_rmf, search_tol_da, max_results
            )

            nl_mass = precursor_neutral_mass - frag_neutral
            nl_formulas = (
                self._search(nl_mass, mol_elems, mol_order, mol_rmf, search_tol_da, max_results)
                if nl_mass > 0
                else []
            )

            # Re-express ppm errors relative to precursor m/z so they are
            # always consistent with the user-set threshold.
            if ref_for_ppm is not None:
                frag_formulas = [
                    (f, m, (m - frag_neutral) / ref_for_ppm * 1e6)
                    for f, m, _ in frag_formulas
                ]
                nl_formulas = [
                    (f, m, (m - nl_mass) / ref_for_ppm * 1e6)
                    for f, m, _ in nl_formulas
                ]

            results.append(
                {
                    "mz": mz_f,
                    "fragment_formulas": frag_formulas,
                    "neutral_loss_formulas": nl_formulas,
                }
            )
        return results

    def _apply_adduct_elements(self, base_elems, adduct_info):
        """Return a new element-count dict expanded by adduct ElementsAdded/Lost."""
        import pandas as pd

        def _get_str(key):
            val = adduct_info.get(key) if hasattr(adduct_info, "get") else None
            if val is None:
                try:
                    val = adduct_info[key]
                except (KeyError, IndexError, TypeError):
                    val = None
            try:
                if val is not None and pd.isna(val):
                    val = None
            except Exception:
                pass
            return val.strip() if isinstance(val, str) and val.strip() else None

        result = dict(base_elems)

        added_str = _get_str("ElementsAdded")
        if added_str:
            added = {k: v for k, v in self.ft.parseFormula(added_str).items() if not self.ft.isIso(k)}
            for elem, cnt in added.items():
                result[elem] = result.get(elem, 0) + cnt

        lost_str = _get_str("ElementsLost")
        if lost_str:
            lost = {k: v for k, v in self.ft.parseFormula(lost_str).items() if not self.ft.isIso(k)}
            for elem, cnt in lost.items():
                result[elem] = max(0, result.get(elem, 0) - cnt)

        return {k: v for k, v in result.items() if v > 0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search(self, target_mass, max_elems, elem_order, remaining_mass_from, tol_da, max_results):
        """Return up to *max_results* formula candidates matching *target_mass*."""
        if target_mass <= 0:
            return []
        candidates = []
        self._dfs(
            target_mass, tol_da, max_elems, elem_order, remaining_mass_from,
            0, {}, 0.0, candidates, max_results,
        )
        candidates.sort(key=lambda x: abs(x[2]))
        return candidates

    def _dfs(self, target, tol_da, max_elems, elem_order, rmf,
             depth, current, mass_so_far, candidates, max_results):
        """Depth-first search over element counts with mass-based pruning."""
        if len(candidates) >= max_results:
            return

        if depth == len(elem_order):
            diff = mass_so_far - target
            if abs(diff) <= tol_da:
                ppm_err = diff / target * 1e6
                formula_elems = {k: v for k, v in current.items() if v > 0}
                formula_str = self._formula_to_str(formula_elems)
                if formula_str:
                    candidates.append((formula_str, mass_so_far, ppm_err))
            return

        elem = elem_order[depth]
        e_mass = self.ft.elemDetails[elem][3]
        max_cnt = max_elems[elem]
        remaining = rmf[depth + 1]  # max further mass from elements at depth+1 ..

        for cnt in range(max_cnt + 1):
            partial = mass_so_far + cnt * e_mass

            # Prune: already heavier than target + tolerance
            if partial > target + tol_da:
                break

            # Prune: even using all remaining elements at max we can't reach the target
            if partial + remaining < target - tol_da:
                continue

            current[elem] = cnt
            self._dfs(
                target, tol_da, max_elems, elem_order, rmf,
                depth + 1, current, partial, candidates, max_results,
            )

        current.pop(elem, None)

    @staticmethod
    def _formula_to_str(elems):
        """Return a Hill-notation formula string (C first, H second, then alphabetical).

        Zero-count elements are silently skipped.
        """
        elems = {k: v for k, v in elems.items() if v > 0}
        if not elems:
            return ""
        keys = sorted(elems.keys())
        ordered = []
        for priority in ("C", "H"):
            if priority in keys:
                ordered.append(priority)
                keys.remove(priority)
        ordered.extend(keys)
        parts = []
        for e in ordered:
            c = elems[e]
            parts.append(e + (str(c) if c > 1 else ""))
        return "".join(parts)
