# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.8
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_HWsim', [dirname(__file__)])
        except ImportError:
            import _HWsim
            return _HWsim
        if fp is not None:
            try:
                _mod = imp.load_module('_HWsim', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _HWsim = swig_import_helper()
    del swig_import_helper
else:
    import _HWsim
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0


class FunctionUnit(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FunctionUnit, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FunctionUnit, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = _HWsim.new_FunctionUnit()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _HWsim.delete_FunctionUnit
    __del__ = lambda self: None

    def PrintProperty(self, str):
        return _HWsim.FunctionUnit_PrintProperty(self, str)

    def SaveOutput(self, str, outputFile):
        return _HWsim.FunctionUnit_SaveOutput(self, str, outputFile)

    def MagicLayout(self):
        return _HWsim.FunctionUnit_MagicLayout(self)

    def OverrideLayout(self):
        return _HWsim.FunctionUnit_OverrideLayout(self)
    __swig_setmethods__["height"] = _HWsim.FunctionUnit_height_set
    __swig_getmethods__["height"] = _HWsim.FunctionUnit_height_get
    if _newclass:
        height = _swig_property(_HWsim.FunctionUnit_height_get, _HWsim.FunctionUnit_height_set)
    __swig_setmethods__["width"] = _HWsim.FunctionUnit_width_set
    __swig_getmethods__["width"] = _HWsim.FunctionUnit_width_get
    if _newclass:
        width = _swig_property(_HWsim.FunctionUnit_width_get, _HWsim.FunctionUnit_width_set)
    __swig_setmethods__["area"] = _HWsim.FunctionUnit_area_set
    __swig_getmethods__["area"] = _HWsim.FunctionUnit_area_get
    if _newclass:
        area = _swig_property(_HWsim.FunctionUnit_area_get, _HWsim.FunctionUnit_area_set)
    __swig_setmethods__["emptyArea"] = _HWsim.FunctionUnit_emptyArea_set
    __swig_getmethods__["emptyArea"] = _HWsim.FunctionUnit_emptyArea_get
    if _newclass:
        emptyArea = _swig_property(_HWsim.FunctionUnit_emptyArea_get, _HWsim.FunctionUnit_emptyArea_set)
    __swig_setmethods__["usedArea"] = _HWsim.FunctionUnit_usedArea_set
    __swig_getmethods__["usedArea"] = _HWsim.FunctionUnit_usedArea_get
    if _newclass:
        usedArea = _swig_property(_HWsim.FunctionUnit_usedArea_get, _HWsim.FunctionUnit_usedArea_set)
    __swig_setmethods__["totalArea"] = _HWsim.FunctionUnit_totalArea_set
    __swig_getmethods__["totalArea"] = _HWsim.FunctionUnit_totalArea_get
    if _newclass:
        totalArea = _swig_property(_HWsim.FunctionUnit_totalArea_get, _HWsim.FunctionUnit_totalArea_set)
    __swig_setmethods__["readLatency"] = _HWsim.FunctionUnit_readLatency_set
    __swig_getmethods__["readLatency"] = _HWsim.FunctionUnit_readLatency_get
    if _newclass:
        readLatency = _swig_property(_HWsim.FunctionUnit_readLatency_get, _HWsim.FunctionUnit_readLatency_set)
    __swig_setmethods__["writeLatency"] = _HWsim.FunctionUnit_writeLatency_set
    __swig_getmethods__["writeLatency"] = _HWsim.FunctionUnit_writeLatency_get
    if _newclass:
        writeLatency = _swig_property(_HWsim.FunctionUnit_writeLatency_get, _HWsim.FunctionUnit_writeLatency_set)
    __swig_setmethods__["readDynamicEnergy"] = _HWsim.FunctionUnit_readDynamicEnergy_set
    __swig_getmethods__["readDynamicEnergy"] = _HWsim.FunctionUnit_readDynamicEnergy_get
    if _newclass:
        readDynamicEnergy = _swig_property(_HWsim.FunctionUnit_readDynamicEnergy_get, _HWsim.FunctionUnit_readDynamicEnergy_set)
    __swig_setmethods__["writeDynamicEnergy"] = _HWsim.FunctionUnit_writeDynamicEnergy_set
    __swig_getmethods__["writeDynamicEnergy"] = _HWsim.FunctionUnit_writeDynamicEnergy_get
    if _newclass:
        writeDynamicEnergy = _swig_property(_HWsim.FunctionUnit_writeDynamicEnergy_get, _HWsim.FunctionUnit_writeDynamicEnergy_set)
    __swig_setmethods__["leakage"] = _HWsim.FunctionUnit_leakage_set
    __swig_getmethods__["leakage"] = _HWsim.FunctionUnit_leakage_get
    if _newclass:
        leakage = _swig_property(_HWsim.FunctionUnit_leakage_get, _HWsim.FunctionUnit_leakage_set)
    __swig_setmethods__["newWidth"] = _HWsim.FunctionUnit_newWidth_set
    __swig_getmethods__["newWidth"] = _HWsim.FunctionUnit_newWidth_get
    if _newclass:
        newWidth = _swig_property(_HWsim.FunctionUnit_newWidth_get, _HWsim.FunctionUnit_newWidth_set)
    __swig_setmethods__["newHeight"] = _HWsim.FunctionUnit_newHeight_set
    __swig_getmethods__["newHeight"] = _HWsim.FunctionUnit_newHeight_get
    if _newclass:
        newHeight = _swig_property(_HWsim.FunctionUnit_newHeight_get, _HWsim.FunctionUnit_newHeight_set)
    __swig_setmethods__["readPower"] = _HWsim.FunctionUnit_readPower_set
    __swig_getmethods__["readPower"] = _HWsim.FunctionUnit_readPower_get
    if _newclass:
        readPower = _swig_property(_HWsim.FunctionUnit_readPower_get, _HWsim.FunctionUnit_readPower_set)
    __swig_setmethods__["writePower"] = _HWsim.FunctionUnit_writePower_set
    __swig_getmethods__["writePower"] = _HWsim.FunctionUnit_writePower_get
    if _newclass:
        writePower = _swig_property(_HWsim.FunctionUnit_writePower_get, _HWsim.FunctionUnit_writePower_set)
FunctionUnit_swigregister = _HWsim.FunctionUnit_swigregister
FunctionUnit_swigregister(FunctionUnit)

class SwigPyIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SwigPyIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SwigPyIterator, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _HWsim.delete_SwigPyIterator
    __del__ = lambda self: None

    def value(self):
        return _HWsim.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _HWsim.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _HWsim.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _HWsim.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _HWsim.SwigPyIterator_equal(self, x)

    def copy(self):
        return _HWsim.SwigPyIterator_copy(self)

    def next(self):
        return _HWsim.SwigPyIterator_next(self)

    def __next__(self):
        return _HWsim.SwigPyIterator___next__(self)

    def previous(self):
        return _HWsim.SwigPyIterator_previous(self)

    def advance(self, n):
        return _HWsim.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _HWsim.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _HWsim.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _HWsim.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _HWsim.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _HWsim.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _HWsim.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self
SwigPyIterator_swigregister = _HWsim.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

class HWsim(FunctionUnit):
    __swig_setmethods__ = {}
    for _s in [FunctionUnit]:
        __swig_setmethods__.update(getattr(_s, '__swig_setmethods__', {}))
    __setattr__ = lambda self, name, value: _swig_setattr(self, HWsim, name, value)
    __swig_getmethods__ = {}
    for _s in [FunctionUnit]:
        __swig_getmethods__.update(getattr(_s, '__swig_getmethods__', {}))
    __getattr__ = lambda self, name: _swig_getattr(self, HWsim, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = _HWsim.new_HWsim()
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _HWsim.delete_HWsim
    __del__ = lambda self: None
    __swig_setmethods__["inputParameter"] = _HWsim.HWsim_inputParameter_set
    __swig_getmethods__["inputParameter"] = _HWsim.HWsim_inputParameter_get
    if _newclass:
        inputParameter = _swig_property(_HWsim.HWsim_inputParameter_get, _HWsim.HWsim_inputParameter_set)
    __swig_setmethods__["tech"] = _HWsim.HWsim_tech_set
    __swig_getmethods__["tech"] = _HWsim.HWsim_tech_get
    if _newclass:
        tech = _swig_property(_HWsim.HWsim_tech_get, _HWsim.HWsim_tech_set)
    __swig_setmethods__["cell"] = _HWsim.HWsim_cell_set
    __swig_getmethods__["cell"] = _HWsim.HWsim_cell_get
    if _newclass:
        cell = _swig_property(_HWsim.HWsim_cell_get, _HWsim.HWsim_cell_set)
    __swig_setmethods__["params"] = _HWsim.HWsim_params_set
    __swig_getmethods__["params"] = _HWsim.HWsim_params_get
    if _newclass:
        params = _swig_property(_HWsim.HWsim_params_get, _HWsim.HWsim_params_set)

    def PrintProperty(self):
        return _HWsim.HWsim_PrintProperty(self)

    def SaveOutput(self, CoreIndex, outputFile):
        return _HWsim.HWsim_SaveOutput(self, CoreIndex, outputFile)

    def Initialize(self):
        return _HWsim.HWsim_Initialize(self)

    def CalculateArea(self):
        return _HWsim.HWsim_CalculateArea(self)

    def CalculateLatency(self):
        return _HWsim.HWsim_CalculateLatency(self)

    def CalculatePower(self, activityRowRead):
        return _HWsim.HWsim_CalculatePower(self, activityRowRead)
    __swig_setmethods__["initialized"] = _HWsim.HWsim_initialized_set
    __swig_getmethods__["initialized"] = _HWsim.HWsim_initialized_get
    if _newclass:
        initialized = _swig_property(_HWsim.HWsim_initialized_get, _HWsim.HWsim_initialized_set)
    __swig_setmethods__["numOutput"] = _HWsim.HWsim_numOutput_set
    __swig_getmethods__["numOutput"] = _HWsim.HWsim_numOutput_get
    if _newclass:
        numOutput = _swig_property(_HWsim.HWsim_numOutput_get, _HWsim.HWsim_numOutput_set)
    __swig_setmethods__["numArrayCol"] = _HWsim.HWsim_numArrayCol_set
    __swig_getmethods__["numArrayCol"] = _HWsim.HWsim_numArrayCol_get
    if _newclass:
        numArrayCol = _swig_property(_HWsim.HWsim_numArrayCol_get, _HWsim.HWsim_numArrayCol_set)
    __swig_setmethods__["numArrayRow"] = _HWsim.HWsim_numArrayRow_set
    __swig_getmethods__["numArrayRow"] = _HWsim.HWsim_numArrayRow_get
    if _newclass:
        numArrayRow = _swig_property(_HWsim.HWsim_numArrayRow_get, _HWsim.HWsim_numArrayRow_set)
    __swig_setmethods__["numCellPerSynapse"] = _HWsim.HWsim_numCellPerSynapse_set
    __swig_getmethods__["numCellPerSynapse"] = _HWsim.HWsim_numCellPerSynapse_get
    if _newclass:
        numCellPerSynapse = _swig_property(_HWsim.HWsim_numCellPerSynapse_get, _HWsim.HWsim_numCellPerSynapse_set)
    __swig_setmethods__["numBitInput"] = _HWsim.HWsim_numBitInput_set
    __swig_getmethods__["numBitInput"] = _HWsim.HWsim_numBitInput_get
    if _newclass:
        numBitInput = _swig_property(_HWsim.HWsim_numBitInput_get, _HWsim.HWsim_numBitInput_set)
    __swig_setmethods__["numWeightBit"] = _HWsim.HWsim_numWeightBit_set
    __swig_getmethods__["numWeightBit"] = _HWsim.HWsim_numWeightBit_get
    if _newclass:
        numWeightBit = _swig_property(_HWsim.HWsim_numWeightBit_get, _HWsim.HWsim_numWeightBit_set)
    __swig_setmethods__["numCellBit"] = _HWsim.HWsim_numCellBit_set
    __swig_getmethods__["numCellBit"] = _HWsim.HWsim_numCellBit_get
    if _newclass:
        numCellBit = _swig_property(_HWsim.HWsim_numCellBit_get, _HWsim.HWsim_numCellBit_set)
    __swig_setmethods__["numSABit"] = _HWsim.HWsim_numSABit_set
    __swig_getmethods__["numSABit"] = _HWsim.HWsim_numSABit_get
    if _newclass:
        numSABit = _swig_property(_HWsim.HWsim_numSABit_get, _HWsim.HWsim_numSABit_set)
    __swig_setmethods__["numBitPartialSum"] = _HWsim.HWsim_numBitPartialSum_set
    __swig_getmethods__["numBitPartialSum"] = _HWsim.HWsim_numBitPartialSum_get
    if _newclass:
        numBitPartialSum = _swig_property(_HWsim.HWsim_numBitPartialSum_get, _HWsim.HWsim_numBitPartialSum_set)
    __swig_setmethods__["wireWidth"] = _HWsim.HWsim_wireWidth_set
    __swig_getmethods__["wireWidth"] = _HWsim.HWsim_wireWidth_get
    if _newclass:
        wireWidth = _swig_property(_HWsim.HWsim_wireWidth_get, _HWsim.HWsim_wireWidth_set)
    __swig_setmethods__["numof1"] = _HWsim.HWsim_numof1_set
    __swig_getmethods__["numof1"] = _HWsim.HWsim_numof1_get
    if _newclass:
        numof1 = _swig_property(_HWsim.HWsim_numof1_get, _HWsim.HWsim_numof1_set)
    __swig_setmethods__["numof2"] = _HWsim.HWsim_numof2_set
    __swig_getmethods__["numof2"] = _HWsim.HWsim_numof2_get
    if _newclass:
        numof2 = _swig_property(_HWsim.HWsim_numof2_get, _HWsim.HWsim_numof2_set)
    __swig_setmethods__["numof3"] = _HWsim.HWsim_numof3_set
    __swig_getmethods__["numof3"] = _HWsim.HWsim_numof3_get
    if _newclass:
        numof3 = _swig_property(_HWsim.HWsim_numof3_get, _HWsim.HWsim_numof3_set)
    __swig_setmethods__["numof4"] = _HWsim.HWsim_numof4_set
    __swig_getmethods__["numof4"] = _HWsim.HWsim_numof4_get
    if _newclass:
        numof4 = _swig_property(_HWsim.HWsim_numof4_get, _HWsim.HWsim_numof4_set)
    __swig_setmethods__["numof5"] = _HWsim.HWsim_numof5_set
    __swig_getmethods__["numof5"] = _HWsim.HWsim_numof5_get
    if _newclass:
        numof5 = _swig_property(_HWsim.HWsim_numof5_get, _HWsim.HWsim_numof5_set)
    __swig_setmethods__["numof6"] = _HWsim.HWsim_numof6_set
    __swig_getmethods__["numof6"] = _HWsim.HWsim_numof6_get
    if _newclass:
        numof6 = _swig_property(_HWsim.HWsim_numof6_get, _HWsim.HWsim_numof6_set)
    __swig_setmethods__["numof7"] = _HWsim.HWsim_numof7_set
    __swig_getmethods__["numof7"] = _HWsim.HWsim_numof7_get
    if _newclass:
        numof7 = _swig_property(_HWsim.HWsim_numof7_get, _HWsim.HWsim_numof7_set)
    __swig_setmethods__["numof8"] = _HWsim.HWsim_numof8_set
    __swig_getmethods__["numof8"] = _HWsim.HWsim_numof8_get
    if _newclass:
        numof8 = _swig_property(_HWsim.HWsim_numof8_get, _HWsim.HWsim_numof8_set)
    __swig_setmethods__["numof9"] = _HWsim.HWsim_numof9_set
    __swig_getmethods__["numof9"] = _HWsim.HWsim_numof9_get
    if _newclass:
        numof9 = _swig_property(_HWsim.HWsim_numof9_get, _HWsim.HWsim_numof9_set)
    __swig_setmethods__["numof10"] = _HWsim.HWsim_numof10_set
    __swig_getmethods__["numof10"] = _HWsim.HWsim_numof10_get
    if _newclass:
        numof10 = _swig_property(_HWsim.HWsim_numof10_get, _HWsim.HWsim_numof10_set)
    __swig_setmethods__["core"] = _HWsim.HWsim_core_set
    __swig_getmethods__["core"] = _HWsim.HWsim_core_get
    if _newclass:
        core = _swig_property(_HWsim.HWsim_core_get, _HWsim.HWsim_core_set)
HWsim_swigregister = _HWsim.HWsim_swigregister
HWsim_swigregister(HWsim)

# This file is compatible with both classic and new-style classes.


