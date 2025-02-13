function logits = inv_affine_sigmoid(probs, lo, hi)
    % The inverse of affine_sigmoid(., lo, hi).
    if ~(lo < hi)
        error('`lo` (%g) must be < `hi` (%g)', lo, hi);
    end

    logits = logit((probs - lo) / (hi - lo));
end
