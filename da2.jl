#DA Algorithm
#Todo: Algorithm2, alg to list all stable matchings, R_men, R_women

using DataStructures
using RCall
using Distributions
using PyPlot

set_x_as(n) = rand(n)
set_x_ds(m) = rand(m)
utility(beta, gamma, i, j, f_x_as, m_x_ds, f_x_ds, epsilons) = beta*f_x_as[j] - gamma*(m_x_ds[i] - f_x_ds[j])^2 + epsilons[i, j]
function set_epsilons(m, n)
    #d = LogNormal()
    #return reshape(rand(d, m*n), m, n)
    return reshape([logistic() for i in 1:m*n], m, n)
end

function logistic()
    u = rand()
    return log(u/(1-u))
end

function get_mn_prefs_by_utility(m, n, beta, gamma, m_x_as, f_x_as, m_x_ds, f_x_ds, epsilons)
    m_prefs = Array(Int, n+1, m)
    f_prefs = Array(Int, m+1, n)

    for i in 1:m
        m_prefs[1:(end-1), i] = sort(1:n, by = j -> utility(beta, gamma, i, j, f_x_as, m_x_ds, f_x_ds, epsilons), rev=true)
        m_prefs[end, i] = 0
    end
    for j in 1:n
        f_prefs[1:(end-1), j] = sort(1:m, by = i -> utility(beta, gamma, j, i, m_x_as, f_x_ds, m_x_ds, transpose(epsilons)), rev=true)
        f_prefs[end, j] = 0
    end
    return m_prefs, f_prefs
end

function count_stable_matchings_fast(m_prefs, f_prefs)
    R"get_ranks <- function(prefs, include_self = TRUE) {
    if (include_self == TRUE) {
       ranks <- array(rep(NA, (nrow(prefs)-1)*ncol(prefs)), dim = c((nrow(prefs)-1), ncol(prefs)))
    } else {
       ranks <- array(rep(NA, nrow(prefs)*ncol(prefs)), dim = c(nrow(prefs), ncol(prefs)))
    }
    for (j in 1:ncol(prefs)) {
       c = 1
       na = FALSE
       for (i in prefs[, j]) {
           if (i == 0) {
               na = TRUE
           }
           if (na == TRUE) {
               ranks[i, j] = NA
           } else {
               ranks[i, j] = c
           }

           c = c + 1
       }
    }
    return(ranks)
    }"
    R"library(matchingMarkets)"
    _, ms, _ = rcopy(R"hri(s.prefs = get_ranks($m_prefs), c.prefs = get_ranks($f_prefs))")
    return ms[2][end, 1]
end

function pmm_fast(m_prefs, f_prefs)
    R"get_ranks <- function(prefs, include_self = TRUE) {
    if (include_self == TRUE) {
       ranks <- array(rep(NA, (nrow(prefs)-1)*ncol(prefs)), dim = c((nrow(prefs)-1), ncol(prefs)))
    } else {
       ranks <- array(rep(NA, nrow(prefs)*ncol(prefs)), dim = c(nrow(prefs), ncol(prefs)))
    }
    for (j in 1:ncol(prefs)) {
       c = 1
       na = FALSE
       for (i in prefs[, j]) {
           if (i == 0) {
               na = TRUE
           }
           if (na == TRUE) {
               ranks[i, j] = NA
           } else {
               ranks[i, j] = c
           }

           c = c + 1
       }
    }
    return(ranks)
    }"
    R"library(matchingMarkets)"
    _, ms, _ = rcopy(R"hri(s.prefs = get_ranks($m_prefs), c.prefs = get_ranks($f_prefs))")

    m_stable_partner_list = [[] for i in 1:size(m_prefs, 2)]

    for i in 1:size(ms[2], 1)
        push!(m_stable_partner_list[ms[2][i, 3]], ms[2][i, 2])
    end

    return length(filter(x -> x > 1, map(length, m_stable_partner_list)))/size(m_prefs, 2)
end

function call_match{T <: Integer}(m_prefs::Array{T, 2}, f_prefs::Array{T, 2})
    m::Int = size(m_prefs, 2)
    n::Int = size(f_prefs, 2)

    f_ranks = get_ranks(f_prefs)
    m_pointers = zeros(Int, m)
    f_matched = zeros(Int, n)

    m_matched_tf = falses(m)
    m_offers = zeros(Int, 2, m+1)
    m_offers[1, 1] = 1

    da_match(m, n, f_ranks, m_prefs, f_prefs, m_pointers, f_matched, m_matched_tf, m_offers)
    return convert_pointer_to_list(m, f_matched)
end
call_match_wosm(m_prefs, f_prefs) = reverse(call_match(f_prefs, m_prefs))

@inbounds function get_ranks{T <: Integer}(prefs::Array{T, 2})
    ranks = Array(eltype(prefs), size(prefs))
    for j in 1:size(prefs, 2)
        for (r, i) in enumerate(prefs[:, j])
            if i != 0
                ranks[i, j] = r
            else
                ranks[end, j] = r
            end
        end
    end
    return ranks
end

function convert_pointer_to_list{T <: Integer}(m::Int, f_matched::Array{T, 1})
    m_matched = [findfirst(f_matched, i) for i in 1:m]
    return m_matched, f_matched
end

@inbounds function proceed_pointer!{T <: Integer}(m::Int, n::Int, m_pointers::Array{T, 1}, m_matched_tf, m_prefs)
    for i in 1:m
        if m_pointers[i] > n
            m_matched_tf[i] = true
        else
            if !m_matched_tf[i]
                m_pointers[i] += 1
                if m_prefs[m_pointers[i], i] == 0
                    m_matched_tf[i] = true
                end
            end
        end
    end
end

@inbounds function create_offers!{T <: Integer}(m::Int, m_prefs::Array{T, 2}, m_matched_tf, m_pointers::Array{T, 1}, m_offers)
    c::Int = 1
    for i in 1:m
        if !m_matched_tf[i] && m_prefs[m_pointers[i], i] != 0
            m_offers[1, c] = i
            m_offers[2, c] = m_prefs[m_pointers[i], i]
            c += 1
        end
    end
    m_offers[1, c] = 0
    m_offers[2, c] = 0
end

@inbounds function decide_to_accept!{T <: Integer}(f_matched::Array{T, 1}, f_ranks::Array{T, 2}, f_prefs::Array{T, 2}, m_offers, m_matched_tf)
    for k in 1:length(m_offers)
        m_offers[1, k] == 0 && break
        if f_matched[m_offers[2, k]] == 0
            if f_ranks[end, m_offers[2, k]] > f_ranks[m_offers[1, k], m_offers[2, k]]
                f_matched[m_offers[2, k]] = m_offers[1, k]
                m_matched_tf[m_offers[1, k]] = true
            end
        else
            if f_ranks[f_matched[m_offers[2, k]], m_offers[2, k]] > f_ranks[m_offers[1, k], m_offers[2, k]]
                m_matched_tf[f_matched[m_offers[2, k]]] = false
                f_matched[m_offers[2, k]] = m_offers[1, k]
                m_matched_tf[m_offers[1, k]] = true
            end
        end
    end
end

function da_match{T <: Integer}(m::Int, n::Int, f_ranks::Array{T, 2}, m_prefs::Array{T, 2}, f_prefs::Array{T, 2}, m_pointers::Array{T, 1}, f_matched::Array{T, 1}, m_matched_tf, m_offers)
    while m_offers[1, 1] != 0
        proceed_pointer!(m, n, m_pointers, m_matched_tf, m_prefs)
        create_offers!(m, m_prefs, m_matched_tf, m_pointers, m_offers)
        decide_to_accept!(f_matched, f_ranks, f_prefs, m_offers, m_matched_tf)
    end
end

#####functions for debug#####


function is_stable_matching(m_matched, f_matched, m_prefs, f_prefs)#########self matched no toki stability
    for (i, j) in enumerate(m_matched)
        index_of_j = findfirst(m_prefs[:, i], j)
        if index_of_j > 1
            for k in 1:(index_of_j-1)
                better_j = m_prefs[k, i]#yori yoi female
                better_j == 0 && continue
                index_of_i_by_better_j = findfirst(f_prefs[:, better_j], f_matched[better_j])#better_j no match shita aite no index
                if index_of_i_by_better_j > 1
                    if in(i, f_prefs[:, better_j][1:(index_of_i_by_better_j-1)])
                        return false
                    end
                end
            end
        end
    end
    return true
end

function is_stable_matching(m_matched, m_prefs, f_prefs)
    _, f_matched = reverse(convert_pointer_to_list(size(f_prefs, 2), m_matched))
    for (i, j) in enumerate(m_matched)
        index_of_j = findfirst(m_prefs[:, i], j)
        if index_of_j > 1
            for k in 1:(index_of_j-1)
                better_j = m_prefs[k, i]#yori yoi female
                better_j == 0 && continue
                index_of_i_by_better_j = findfirst(f_prefs[:, better_j], f_matched[better_j])#better_j no match shita aite no index
                if index_of_i_by_better_j > 1
                    if in(i, f_prefs[:, better_j][1:(index_of_i_by_better_j-1)])
                        return false
                    end
                end
            end
        end
    end
    return true
end

function check_results(m_matched, f_pointers)
    for (i, f) in enumerate(m_matched)
        if f != 0
            f_pointers[f] != i && error("Matching Incomplete with male $i, m_matched[$i] = $(m_matched[i]) though f_pointers[$f] = $(f_pointers[f])")
        elseif f == 0
            in(i, f_pointers) && error("Matching Incomplete with male $i, m_matched[$i] = $(m_matched[i]) though f_pointers[$f] = $(f_pointers[f])")
        end
    end
    for (j, m) in enumerate(f_pointers)
        if m != 0
            m_matched[m] != j && error("Matching Incomplete with female $j, f_pointers[$j] = $(f_pointers[j]) though m_matched[$m] = $(m_matched[m])")
        elseif m == 0
            in(j, m_matched) && error("Matching Incomplete with female $j, f_pointers[$j] = $(f_pointers[j]) though m_matched[$m] = $(m_matched[m])")
        end
    end
    return true
end

"""
function generate_random_preference_data(m, n, one2many = false)
    m_prefs = Array(Int, n+1, m)
    f_prefs = one2many ? Array(Int, m, n) : Array(Int, m+1, n)
    for i in 1:m
        m_prefs[:, i] = shuffle(collect(0:n))
    end
    for j in 1:n
        f_prefs[:, j] = one2many ? shuffle(collect(1:m)) : shuffle(collect(0:m))
    end
    return m_prefs, f_prefs
end
"""

function generate_random_preference_data(m, n; complete = true, stage = 0)
    m_prefs = Array(Int, n+1, m)
    f_prefs = Array(Int, m+1, n)
    if complete
        for i in 1:m
            m_prefs[1:(end-1), i] = shuffle(collect(1:n))
            m_prefs[end, i] = 0
        end
        for j in 1:n
            f_prefs[1:(end-1), j] = shuffle(collect(1:m))
            f_prefs[end, j] = 0
        end
        return m_prefs, f_prefs
    else
        if stage == 0
            for i in 1:m
                m_prefs[:, i] = shuffle(collect(0:n))
            end
            for j in 1:n
                f_prefs[:, j] = shuffle(collect(0:m))
            end
            return m_prefs, f_prefs
        else
            for i in 1:m
                m_prefs[:, i] = insert!(shuffle(collect(1:n)), n-stage+2, 0)
            end
            for j in 1:n
                f_prefs[:, j] = insert!(shuffle(collect(1:m)), m-stage+2, 0)
            end
            return m_prefs, f_prefs
        end
    end
end

function check_data(m_prefs, f_prefs)
    m = size(m_prefs, 2)
    n = size(f_prefs, 2)
    size(m_prefs, 1) != n+1 && error("the size of m_prefs must be (n+1, *)")
    size(f_prefs, 1) != m+1 && error("the size of f_prefs must be (m+1, *)")
    all([Set(m_prefs[:, i]) == Set(0:n) for i in 1:m]) || error("error in m_prefs")
    all([Set(f_prefs[:, j]) == Set(0:m) for j in 1:n]) || error("error in f_prefs")
    return true
end

function create_matching(ns)
    return [collect(n_perm) for n_perm in permutations(ns)]
end

function list_all_matchings(m, n)
    all_matchings = []
    if m > n#################################
        mss = combinations(collect(1:m), m - n)
        ns = collect(1:n)
        matchings = create_matching(ns)
        for ms in mss
            for matching in matchings
                matching_cp = copy(matching)
                for i in ms
                    insert!(matching_cp, i, 0)
                end
                push!(all_matchings, matching_cp)
            end
        end
        return all_matchings
    elseif m < n
        nss = combinations(collect(1:n), m)
        for ns in nss
            append!(all_matchings, create_matching(ns))
        end
        return all_matchings
    else
        return create_matching(collect(1:n))
    end
end

function get_stable_matchings(matchings, m_prefs, f_prefs)
    n = size(f_prefs, 2)
    stable_matchings = []
    for matching in matchings
        _, f_matched = reverse(convert_pointer_to_list(n, matching))
        if is_stable_matching(matching, f_matched, m_prefs, f_prefs)
            push!(stable_matchings, matching)
        end
    end
    return stable_matchings
end

#get_stable_matchings(matchings, m_prefs, f_prefs) = filter(matching -> is_stable_matching(matching, first(convert_pointer_to_list(n, matching)), m_prefs, f_prefs), matchings)

function count_stable_matchings(matchings, m_prefs, f_prefs)
    return length(get_stable_matchings(matchings, m_prefs, f_prefs))
end
"""
function r_men(m_matched, m_prefs)
    sum(map(i -> findfirst(m_prefs[:, i], m_matched[i]), map(first, filter((i, j) -> j != 0, collect(enumerate(m_matched)))))) / (length(filter(j -> j != 0, m_matched))/length(m_matched))
end
"""

function r_men(m_matched, m_prefs)
    matched_num = 0
    s = 0
    for (i, j) in enumerate(m_matched)
        j == 0 && continue
        matched_num += 1
        s += findfirst(m_prefs[:, i], j)
    end
    return s/matched_num
end
r_women = r_men

function pmm(matchings, m_prefs, f_prefs)#Percentage of matched men with multiple stable partner
    c = 0
    stable_matchings = get_stable_matchings(matchings, m_prefs, f_prefs)
    m = size(m_prefs, 2)
    for i in 1:m
        partners = [matching[i] for matching in stable_matchings]
        if length(Set(partners)) > 1
            c += 1
        end
    end
    return c/m
end

"""
function rsd(m, n)
    if m > n
        ns = append!(collect(1:n), zeros(Int, m-n))
        m_matched = shuffle!(ns)
    elseif m < n
        ns = collect(1:n)
        m_matched = shuffle!(ns)[1:m]
    else
        m_matched = shuffle!(collect(1:n))
    end
    return reverse(convert_pointer_to_list(n, m_matched))
end
"""

function rsd(m, n, m_prefs)
    m_matched = zeros(Int, m)
    for i in shuffle(collect(1:m))
        for j in m_prefs[:, i]
            if j == 0
                m_matched[i] = 0
            else
                if !in(j, m_matched)
                    m_matched[i] = j
                    break
                end
            end
        end
    end
    return reverse(convert_pointer_to_list(n, m_matched))
end

function num_all_matchings(m, n)
    if m >= n
        return convert(Integer, factorial(m)/factorial(m-n))
    elseif m < n
        return convert(Integer, factorial(m) * factorial(n)/factorial(n-m)/factorial(m))
    end
end

#function alg2


function call_simple_match{T <: Integer}(m_prefs::Array{T, 2}, f_prefs::Array{T, 2}, m_first = true)
    max = maximum([maximum(m_prefs), maximum(f_prefs)])
    m::Int = size(m_prefs, 2)
    n::Int = size(f_prefs, 2)
    if !m_first
        m, n = n, m
        m_prefs, f_prefs = f_prefs, m_prefs
    end
    m_pointers = zeros(Int, m)
    m_matched_tf = falses(m)
    f_pointers = zeros(Int, n)
    f_ranks = get_ranks(f_prefs)
    j::Int = 0
    while !(all(m_matched_tf) == true)
        proceed_pointer!(m, n, m_pointers, m_matched_tf, m_prefs)
        for i in 1:m
            if !m_matched_tf[i]
                j = m_prefs[m_pointers[i], i]
                j == 0 && continue
                if f_pointers[j] == 0
                    if f_ranks[end, j] > f_ranks[i, j]
                        f_pointers[j] = i
                        m_matched_tf[i] = true
                    end
                else
                    if f_ranks[f_pointers[j], j] > f_ranks[i, j]
                        m_matched_tf[f_pointers[j]] = false
                        f_pointers[j] = i
                        m_matched_tf[i] = true
                    end
                end
            end
        end
    end
    return m_first ? (Int[findfirst(f_pointers, i) for i in 1:m], f_pointers) : (f_pointers, [findfirst(f_pointers, i) for i in 1:m])
end
