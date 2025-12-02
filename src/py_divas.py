from typing import Dict, Optional

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter


class PyDIVAS:
    """
    Python wrapper for the DIVAS R package.
    Allows calling DIVAS functions from Python using rpy2.

    The wrapper automatically handles the conversion between Python's (n_samples, n_features)
    convention and R's (n_features, n_samples) convention. Input your data in standard
    Python format and the wrapper will transpose it appropriately for R.
    """

    def __init__(self):
        """Initialize the R environment and load the DIVAS package."""
        try:
            ro.r("library(DIVAS)")
            self._define_r_function()
            print("DIVAS R package loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load DIVAS R package: {e}")

    def _define_r_function(self):
        """Define the DIVAS_my R function in the R environment."""
        r_code = """
        DIVAS_my <- function(datablock, nsim = 400, iprint = FALSE, colCent = FALSE, rowCent = FALSE, figdir = NULL, seed = NULL){
          nb <- length(datablock)
          dataname <- names(datablock)
          if(is.null(dataname)){
            warning("Input datablock is unnamed, generic names for data blocks generated.")
            dataname <- paste0("Datablock", 1:nb)
          }

          # Some tuning parameters for algorithms
          theta0 <- 45
          optArgin <- list(0.5, 100, 1.05, 50, 1e-3, 1e-3)
          filterPerc <- 1 - (2 / (1 + sqrt(5))) # "Golden Ratio"
          noisepercentile <- rep(0.5, nb)

          rowSpaces <- vector("list", nb)
          for (ib in seq_len(nb)) {
            rowSpaces[[ib]] <- 0
            datablock[[ib]] <- MatCenterJP(datablock[[ib]], colCent, rowCent)
          }
          
          Phase1 <- DJIVESignalExtractJP(
            datablock = datablock, nsim = nsim,
            iplot = FALSE, colCent = colCent, rowCent = rowCent, cull = filterPerc, noisepercentile = noisepercentile,
            seed = seed
          )
          
          # Step 2: Estimate joint and partially joint structure
          Phase2 <- DJIVEJointStrucEstimateJP(
            VBars = Phase1$VBars, UBars = Phase1$UBars, phiBars = Phase1$phiBars, psiBars = Phase1$psiBars,
            rBars = Phase1$rBars, dataname = dataname, iprint = iprint, figdir = figdir
          )

          # Step 3: Reconstruct DJIVE decomposition
          outstruct <- DJIVEReconstructMJ(
            datablock = datablock, dataname = dataname, outMap = Phase2$outMap,
            keyIdxMap = Phase2$keyIdxMap, jointBlockOrder = Phase2$jointBlockOrder, doubleCenter = 0
          )

          outstruct$rBars <- Phase1$rBars
          outstruct$phiBars <- Phase1$phiBars
          outstruct$psiBars <- Phase1$psiBars
          outstruct$VBars <- Phase1$VBars
          outstruct$UBars <- Phase1$UBars
          outstruct$VVHatCacheBars <- Phase1$VVHatCacheBars
          outstruct$UUHatCacheBars <- Phase1$UUHatCacheBars
          outstruct$jointBasisMapRaw <- Phase2$outMap
          outstruct$jointBlockOrder <- Phase2$jointBlockOrder
          
          # Automatically generate keymapname from keymapid
          ids <- as.integer(names(outstruct$keyIdxMap))
          num_blocks <- length(dataname)
          
          keymapname <- sapply(ids, function(id) {
            binary_str <- R.utils::intToBin(id)
            padded_binary_str <- sprintf(paste0("%0", num_blocks, "s"), binary_str)
            binary_chars <- strsplit(padded_binary_str, "")[[1]]
            selected_indices <- which(rev(binary_chars) == '1')
            selected_names <- dataname[selected_indices]
            paste(selected_names, collapse = "+")
          })
          
          names(keymapname) <- names(outstruct$keyIdxMap)
          outstruct$keymapname <- keymapname

          return(outstruct)
        }
        """
        ro.r(r_code)
        print("DIVAS_my function defined in R environment.")

    def run_divas(
        self,
        datablock: Dict[str, np.ndarray],
        nsim: int = 400,
        iprint: bool = False,
        colCent: bool = False,
        rowCent: bool = False,
        figdir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Run DIVAS analysis on multiple data blocks.

        Parameters
        ----------
        datablock : Dict[str, np.ndarray]
            Dictionary of data blocks where keys are block names and values are numpy arrays.
            Each array should be 2D in Python convention: (n_samples, n_features).
            Matrices will be automatically transposed to R convention (n_features, n_samples).
        nsim : int, optional
            Number of simulations for signal extraction (default: 400).
        iprint : bool, optional
            Whether to print progress information (default: False).
        colCent : bool, optional
            Whether to center columns (default: False).
        rowCent : bool, optional
            Whether to center rows (default: False).
        figdir : str, optional
            Directory to save figures (default: None).
        seed : int, optional
            Random seed for reproducibility (default: None).

        Returns
        -------
        Dict
            Dictionary containing DIVAS decomposition results.

        Examples
        --------
        >>> divas = PyDIVAS()
        >>> data = {
        ...     'Block1': np.random.randn(100, 50),  # 100 samples, 50 features
        ...     'Block2': np.random.randn(100, 30)   # 100 samples, 30 features
        ... }
        >>> results = divas.run_divas(data, nsim=200, seed=42)
        """
        # Convert Python dictionary to R named list
        # IMPORTANT: Transpose matrices from Python (n_samples, n_features)
        # to R convention (n_features, n_samples)
        # R DIVAS expects all matrices to have the same number of COLUMNS (samples)
        with localconverter(ro.default_converter + numpy2ri.converter):
            r_datablock = ro.ListVector(
                {
                    name: ro.r.matrix(data.T, nrow=data.shape[1], ncol=data.shape[0])
                    for name, data in datablock.items()
                }
            )

        # Prepare arguments
        kwargs = {
            "datablock": r_datablock,
            "nsim": nsim,
            "iprint": iprint,
            "colCent": colCent,
            "rowCent": rowCent,
        }

        if figdir is not None:
            kwargs["figdir"] = figdir

        if seed is not None:
            kwargs["seed"] = seed

        # Call the R function
        print(f"Running DIVAS with {len(datablock)} data blocks...")
        self.r_result = ro.r["DIVAS_my"](**kwargs)

        # Convert R result to Python dictionary
        self.result = self._convert_r_result(self.r_result)
        print("DIVAS analysis completed.")

        return self.result

    def _convert_r_result(self, r_result) -> Dict:
        """
        Convert R list result to Python dictionary with numpy arrays.

        Parameters
        ----------
        r_result : rpy2.robjects.ListVector
            R list containing DIVAS results.

        Returns
        -------
        Dict
            Python dictionary with converted results.
        """
        result = {}

        for name in r_result.names:
            r_obj = r_result.rx2(name)

            # Convert based on type
            if isinstance(r_obj, ro.vectors.Matrix):
                result[name] = np.array(r_obj)
            elif isinstance(r_obj, ro.vectors.ListVector):
                # Recursively convert nested lists
                result[name] = self._convert_r_list(r_obj)
            elif isinstance(r_obj, ro.vectors.IntVector):
                result[name] = np.array(r_obj, dtype=int)
            elif isinstance(r_obj, ro.vectors.FloatVector):
                result[name] = np.array(r_obj)
            elif isinstance(r_obj, ro.vectors.StrVector):
                result[name] = list(r_obj)
            else:
                # Keep as is if type is unknown
                result[name] = r_obj

        return result

    def _convert_r_list(self, r_list):
        """Convert R list to Python dictionary or list."""
        # Check if r_list.names exists and is not NULL
        try:
            names = r_list.names
            # In rpy2, NULL names are represented as rpy2.rinterface.NULL
            # which is not None but evaluates to False in boolean context
            if names is not None and names is not ro.NULL:
                # Try to iterate - if it fails, treat as unnamed list
                try:
                    names_list = list(names)
                    if names_list:
                        # Named list -> dictionary
                        return {
                            name: self._convert_r_object(r_list.rx2(name))
                            for name in names_list
                        }
                except (TypeError, AttributeError):
                    pass
        except (AttributeError, TypeError):
            pass

        # Unnamed list or failed to get names -> convert to list
        try:
            return [self._convert_r_object(item) for item in r_list]
        except (TypeError, AttributeError):
            # If we can't iterate, return as is
            return r_list

    def _convert_r_object(self, r_obj):
        """Convert a single R object to Python equivalent."""
        # Handle NULL values
        if r_obj is ro.NULL or r_obj is None:
            return None

        # Handle different R object types
        if isinstance(r_obj, ro.vectors.Matrix):
            return np.array(r_obj)
        elif isinstance(r_obj, ro.vectors.ListVector):
            return self._convert_r_list(r_obj)
        elif isinstance(r_obj, ro.vectors.IntVector):
            arr = np.array(r_obj, dtype=int)
            # Return scalar if single element
            return arr.item() if arr.size == 1 else arr
        elif isinstance(r_obj, ro.vectors.FloatVector):
            arr = np.array(r_obj)
            # Return scalar if single element
            return arr.item() if arr.size == 1 else arr
        elif isinstance(r_obj, ro.vectors.StrVector):
            str_list = list(r_obj)
            # Return single string if only one element
            return str_list[0] if len(str_list) == 1 else str_list
        elif isinstance(r_obj, ro.vectors.BoolVector):
            bool_arr = np.array(r_obj, dtype=bool)
            return bool_arr.item() if bool_arr.size == 1 else bool_arr
        else:
            # For unknown types, try to convert or return as is
            try:
                return np.array(r_obj)
            except Exception:
                return r_obj

    def transform(self, input: np.ndarray, block_identifier: str, data_block_n: int):
        """
        Transform new data into the DIVAS joint space using learned results - multiply by corresponding loadings.

        Parameters
        ----------
        input : np.ndarray
            New data matrix in Python convention (n_samples, n_features).
        results : Dict
            DIVAS results dictionary obtained from run_divas.
        block_identifier : str
            Identifier of the data block to transform - str in decimal defining the combination in binary ('3' = 11).
        data_block_n : int
            Modality index corresponding to the data block.

        Returns
        -------
        np.ndarray
            Transformed data in the joint space.
        """
        # Check that the input data has the correct shape
        if input.ndim != 2:
            raise ValueError("Input data must be a 2D numpy array.")

        # Get loadings for the specified block
        loadings = self.result["matLoadings"][data_block_n - 1][
            block_identifier
        ]  # Adjust for 0-based index

        if input.shape[1] != loadings.shape[0]:
            raise ValueError(
                f"Input data shape ({input.shape}) does not have correct number of features (columns) to match loadings shape ({loadings.shape[0]})."
            )

        # Transform input data: (n_samples, n_features) @ (n_features, n_components) -> (n_samples, n_components)
        transformed = input @ loadings
        return transformed

    def _define_Reconstruct_function(self):
        """Define the DIVAS_reconstruct_loadings R function in the R environment."""
        r_code = """
        outstruct <- DJIVEReconstructMJ(datablock = datablock, dataname = dataname, outMap)
        """
        ro.r(r_code)
        print("DIVAS_reconstruct_loadings function defined in R environment.")

    def reconstruct_loadings(self, new_data: np.ndarray):
        """
        Reconstruct original data from joint space using loadings.

        Parameters
        ----------
        new_data : np.ndarray
            Data in joint space (n_samples, n_components).
        results : Dict
            DIVAS results dictionary obtained from run_divas.
        block_identifier : str
            Identifier of the data block to reconstruct - str in decimal defining the combination in binary ('3' = 11).
        data_block_n : int
            Modality index corresponding to the data block.
        """
        try:
            ro.r("library(DIVAS)")
            print("DIVAS R package loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load DIVAS R package: {e}")

        with localconverter(ro.default_converter + numpy2ri.converter):
            r_datablock = ro.ListVector(
                {
                    name: ro.r.matrix(data.T, nrow=data.shape[1], ncol=data.shape[0])
                    for name, data in new_data.items()
                }
            )

        with localconverter(ro.default_converter + numpy2ri.converter):
            outMap = ro.ListVector(
                {
                    name: ro.r.matrix(data, nrow=data.shape[0], ncol=data.shape[1])
                    for name, data in self.result["jointBasisMapRaw"].items()
                }
            )

        # Convert keyIdxMap: dictionary with string keys and integer/array values
        # Python: {'3': array([1, 2]), '1': 1, '2': 2}
        # R: list('3' = c(1L, 2L), '1' = 1L, '2' = 2L)
        keyIdxMap_dict = {}
        for name, data in self.result["keyIdxMap"].items():
            if isinstance(data, np.ndarray):
                # Convert numpy array to list then to IntVector
                keyIdxMap_dict[name] = ro.IntVector(data.tolist())
            elif isinstance(data, (list, tuple)):
                # Already a list/tuple
                keyIdxMap_dict[name] = ro.IntVector(data)
            else:
                # Scalar value - wrap in a list
                keyIdxMap_dict[name] = ro.IntVector([int(data)])

        keyIdxMap = ro.ListVector(keyIdxMap_dict)

        # Convert jointBlockOrder: list of strings to unnamed R list with character[1] entries
        # Python: ['3', '1', '2']
        # R: list('3', '1', '2') where each element is character[1]
        # Accessed in R as [[1]], [[2]], [[3]]
        # Create as a list of tuples (None, value) for unnamed list
        # jointBlockOrder_items = [(None, ro.StrVector([str(item)])) for item in self.result['jointBlockOrder']]
        # jointBlockOrder_r = ro.ListVector(jointBlockOrder_items)
        jointBlockOrder_r = ro.ListVector(self.result["jointBlockOrder"])

        dataname = list(new_data.keys())
        # dataname = [f'Block{i+1}' for i in range(len(new_data))]

        kwargs = {
            "datablock": r_datablock,
            "dataname": dataname,
            "outMap": outMap,
            "keyIdxMap": keyIdxMap,
            "jointBlockOrder": jointBlockOrder_r,
            "doubleCenter": 0,
        }

        print(f"Running DJIVEReconstruct on new data with {len(new_data)} blocks...")
        outstruct_r = ro.r["DJIVEReconstructMJ"](**kwargs)

        # Convert R result to Python dictionary
        results = self._convert_r_result(outstruct_r)
        print("DJIVEReconstruct completed.")

        return results
