import sys
import numpy as np
import feature_statistics_pb2 as fs


fs_proto = fs.FeatureNameStatistics
histogram_proto = fs.Histogram


def data_type_to_proto_type(dtype):
    dtype = dtype.numpy
    """Converts a Vaex dtype to the FeatureNameStatistics.Type proto enum."""
    if dtype.char in np.typecodes['AllFloat']:
        return fs_proto.FLOAT
    elif (dtype.char in np.typecodes['AllInteger'] or dtype == bool or
          np.issubdtype(dtype, np.datetime64) or
          np.issubdtype(dtype, np.timedelta64)):
        return fs_proto.INT
    else:
        return fs_proto.STRING


def histogram(df, col, n_bins=10, limits=None, limits_finite=None, return_nan=False):    
    if limits:
        x_min, x_max = limits
    else:
        x_min, x_max = df[col][~df[col].isinf()].minmax()
    
    counts = df.count(binby=df[col], shape=n_bins, limits=(x_min, x_max), edges=True)
    num_nan = counts[0]

    # Copy over and under flows.
    counts[-2] += counts[-1]
    counts[2] += counts[1]
    
    # Strip NaN and overflow counts
    counts = counts[2:-1]
    edges = np.linspace(x_min, x_max, n_bins + 1)
    
    # Since we don't have isneginf and isposinf functions.
    if limits_finite is not None:
        if limits_finite[0] is not True:
            edges[0] = float('-inf')
        if limits_finite[1] is not True:
            edges[-1] = float('inf')
    
    if not return_nan:
        return counts, edges
    else:
        return counts, edges, num_nan


def quantile_histogram(df, col, num_bins=10, limits=None):
    quantiles_to_get = [
        x * 100 / num_bins for x in range(num_bins + 1)
    ]
    
    if limits is not None:
        minmax = list(limits)
    else:
        minmax = df[col][~df[col].isinf()].minmax().tolist()

    quantiles = df.percentile_approx(col, quantiles_to_get, percentile_limits=minmax)    
    return quantiles


def proto_from_vaex_dataframe(df, name):
    all_datasets = fs.DatasetFeatureStatisticsList()
    all_datasets.datasets.add(name=name, num_examples=len(df))

    histogram_categorical_levels_count = None

    for col in df.column_names:
        # Fall back to int if no type can be resolved. For example if 
        # all values are missing in a column. 
        try:
            data_type = data_type_to_proto_type(df[col].data_type())
        except:
            data_type = fs_proto.INT
        
        feat = all_datasets.datasets[0].features.add(
            type=data_type, name=col.encode('utf-8')
        )
        
        # Count non-missing 
        non_missing = df[col].count().item()
        missing = len(df) - non_missing
        has_data = (non_missing > 0)
        
        commonstats = None
        # For numeric features, calculate numeric statistics.
        if feat.type in (fs_proto.INT, fs_proto.FLOAT):
            featstats = feat.num_stats
            commonstats = featstats.common_stats
            if has_data:
                featstats.std_dev = df[col].std().item()
                featstats.mean = df[col].mean().item()
                featstats.min = df[col].min().item()
                featstats.max = df[col].max().item()
                featstats.median = df.median_approx(col).item()
                featstats.num_zeros = df.sum(df[col] == 0).item()
                
                # If min and max are inf, then calculate non-inf limits for histogram. 
                # Note: Vaex automatically ignores NANs. 
                hist_min = featstats.min
                hist_max = featstats.max
                num_inf = 0
                if not np.isfinite(hist_min) or not np.isfinite(hist_max):
                    isinf = df[col].isinf()
                    num_inf = isinf.sum().item()
                    hist_min, hist_max = df[col][~isinf].minmax()
                
                limits_finite = np.isfinite([hist_min, hist_max])
                
                counts, edges, num_nan = histogram(df, col, n_bins=10, limits=(hist_min, hist_max), limits_finite=limits_finite, return_nan=True)
                
                hist = featstats.histograms.add()
                hist.type = histogram_proto.STANDARD
                # hist.num_nan = num_nan
                
                for bucket_count in range(len(counts)):
                    bucket = hist.buckets.add(
                        low_value=edges[bucket_count],
                        high_value=edges[bucket_count + 1],
                        sample_count=counts[bucket_count].item())
                
                num_bins = 10
                quantiles = quantile_histogram(df, col, num_bins=num_bins, limits=(hist_min, hist_max))
                quantiles_sample_count = (len(df) - num_inf - num_nan)/num_bins
                
                quant = featstats.histograms.add()
                quant.type = histogram_proto.QUANTILES
                for low, high in zip(quantiles, quantiles[1:]):
                    quant.buckets.add(low_value=low, high_value=high, sample_count=quantiles_sample_count)
        elif feat.type == fs_proto.STRING:
            featstats = feat.string_stats
            commonstats = featstats.common_stats
            if has_data:
                featstats.avg_length = df[col].str.len().mean().item()
                val_counts = df[col].value_counts()
                featstats.unique = len(val_counts)
                sorted_vals = sorted(zip(val_counts.values, val_counts.index.values), reverse=True)
                sorted_vals = sorted_vals[:histogram_categorical_levels_count]
                for val_index, val in enumerate(sorted_vals):
                    try:
                        if (sys.version_info.major < 3 or isinstance(val[1], (bytes, bytearray))):
                            printable_val = val[1].decode('UTF-8', 'strict')
                        else:
                            printable_val = val[1]
                    except(UnicodeDecodeError, UnicodeEncodeError):
                        printable_val = '__BYTES_VALUE__'
                        
                    bucket = featstats.rank_histogram.buckets.add(
                        low_rank=val_index,
                        high_rank=val_index,
                        sample_count=val[0],
                        label=printable_val)
                    if val_index < 2:
                        featstats.top_values.add(value=bucket.label, frequency=bucket.sample_count)
        # Add the common stats regardless of the feature type.
        if has_data:
            commonstats.num_missing = missing
            commonstats.num_non_missing = non_missing
            commonstats.min_num_values = 1
            commonstats.max_num_values = 1
            commonstats.avg_num_values = 1.0
            
            # Create dummy quantile histogram since we are assuming no arrays in a cell. 
            quant = commonstats.num_values_histogram
            quant.type = histogram_proto.QUANTILES
            quantiles = [1.0 for x in range(10 + 1)]
            for low, high in zip(quantiles, quantiles[1:]):
                quant.buckets.add(low_value=low, high_value=high, sample_count=non_missing/10.)
        else:
            commonstats.num_non_missing = 0
            commonstats.num_missing = len(df)

    return all_datasets