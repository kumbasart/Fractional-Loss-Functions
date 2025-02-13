function alpha = affine_sigmoid(logits, lo, hi)
    % Maps reals to (lo, hi), where 0 maps to (lo+hi)/2.
    if ~(lo < hi)
        error('`lo` (%g) must be < `hi` (%g)', lo, hi);
    end
    alpha = sigmoid(logits) * (hi - lo) + lo;
end
