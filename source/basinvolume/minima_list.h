#ifndef _BV_MINIMA_LIST_H__
#define _BV_MINIMA_LIST_H__

#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cmath>

#include "pele/array.h"
#include "pele/distance.h"

namespace bv{

template<typename distance_policy>
class MinimaList: public std::multimap<double,size_t>{
public:
    typedef double energy_t;
    typedef double coor_t;
    typedef size_t index_t;
    typedef std::multimap<coor_t,index_t> map_t;
    typedef distance_policy dist_t;
private:
    const coor_t tol_delta_x;
    const energy_t tol_energy;
    const coor_t tol_delta_x_element;
    std::shared_ptr<dist_t> dist;
    // 1. (minimum_label,delta_x) stored in map (this)
    // 2. energies
    std::vector<energy_t> energy;
    // 3. coordinates: these should be the "alinged coordinates", as obtained in the CheckSameMinimum class before computing the distance to the origin
    std::vector<std::shared_ptr<std::vector<coor_t> > > coor;
    // 4. how many times the minimum has been found
    std::vector<index_t> count;
public:
    MinimaList(const coor_t tol_delta_x_, const energy_t tol_energy_, const coor_t tol_delta_x_element_, std::shared_ptr<dist_t> dist_=NULL);
    /*
     * To make the distance comparison between CheckSameMinimum and here consistent, the tolerances should be set accordingly.
     * */
    index_t nr_distinct_minima()const{return this->size();}
    index_t nr_minimum_visits(const index_t idx)const{return count.at(idx);}
    energy_t get_energy(const index_t idx)const{return energy.at(idx);}
    bool check_new_minimum(const coor_t, const energy_t, pele::Array<coor_t>, pele::Array<coor_t>);
    /*
     * Once the CheckSameMinimum test (given the used tol) has decided that the found minimum is different
     * this function takes the information on that "neighboring" minimum and stores it.
     * If the neighbour was already known, we increase its count and return false.
     * If the neighbour was not yet known (visited), we add it to the list of neighbors and return true.
     * This behaviour is analogous to the behaviour of std::map.
     * We are using std::multimap for now to avoid problems due to floating point arithmetic with the keys (delta_x).
     * Presumably, this is not necessary and can be changed with the typedef above.
     * */
    bool agrees_with_input(const index_t, const energy_t, pele::Array<coor_t>, pele::Array<coor_t>);
    void record_duplicate(const index_t);
    void record_new_minimum(const map_t::const_iterator, const coor_t, const energy_t, pele::Array<double>);
    pele::Array<coor_t> euclidean_displacement_vector(const index_t idx_i, const index_t idx_j, const index_t nr_points)const;
    /*
     * Gives the displacement vector that one should add n_points-1 times to minimum with index j to
     * arrive at minimum i.
     * */
    pele::Array<coor_t> euclidean_displacement_vector(const index_t idx_i, pele::Array<coor_t> coords_j, const index_t nr_points)const;
    /*
     * Same as above, but between minimum with index i and minimum given by coords_j in second argument.
     * This could be used to compute the energy along a displacement line, going from the origin minimum at coords_j to the neighbor at index_i.
     * */
};

template<typename distance_policy>
MinimaList<distance_policy>::MinimaList(const coor_t tol_delta_x_, const energy_t tol_energy_, const coor_t tol_delta_x_element_, std::shared_ptr<dist_t> dist_):
    tol_delta_x(tol_delta_x_), tol_energy(tol_energy_), tol_delta_x_element(tol_delta_x_element_), dist(dist_)
    {
	if (dist==NULL)
	    throw std::runtime_error("MinimaList<distance_policy>::MinimaList: distance policy uninitialised");
    }

template<typename distance_policy>
bool MinimaList<distance_policy>::check_new_minimum(const coor_t delta_x_inp, const energy_t energy_inp, pele::Array<coor_t> coor_inp, pele::Array<coor_t> rattler)
{
    // 1. get possible matches for candidate based on delta_x and dtol
    const map_t::const_iterator low = this->lower_bound(delta_x_inp-tol_delta_x);
    const map_t::const_iterator high= this->upper_bound(delta_x_inp+tol_delta_x);
    // 2. check if candidate agrees with any potential match
    for (map_t::const_iterator i = low; i != high; ++i){
	const index_t this_match = i->second;
	if (agrees_with_input(this_match, energy_inp, coor_inp, rattler)){
	    // candidate new minimum agrees with a previously found one
	    record_duplicate(this_match);
	    return false;
	}
    }
    // candidate new minimum does not agree with any previously found one, store candidate new minimum
    record_new_minimum(low, delta_x_inp, energy_inp, coor_inp);
    return true;
}

template<typename distance_policy>
bool MinimaList<distance_policy>::agrees_with_input(const index_t this_match, const energy_t energy_inp, pele::Array<coor_t> coor_inp, pele::Array<coor_t> rattler)
{
    //1. check for (scalar) delta_x passed
    //2. check: energy match
    if ( fabs(energy.at(this_match)-energy_inp) > tol_energy )
	return false; //failed energy test
    //3. check: coordinate match
    std::vector<coor_t>::const_iterator it = coor.at(this_match)->begin();
    const index_t tmp_N = coor.at(this_match)->size();
    for (index_t i = 0; i < tmp_N; ++i, ++it){
	if ( rattler[i]*fabs( *it - coor_inp[i] ) > tol_delta_x_element )
	    return false; //failed coordinate test
    }
    return true; //all tests passed
}

template<typename distance_policy>
void MinimaList<distance_policy>::record_duplicate(const index_t this_match)
{
    ++count.at(this_match);
}

template<typename distance_policy>
void MinimaList<distance_policy>::record_new_minimum(const map_t::const_iterator insertion_hint, const coor_t delta_x_inp, const energy_t energy_inp, pele::Array<double> coor_inp)
{
    const index_t new_index = nr_distinct_minima();
    this->insert(insertion_hint, std::make_pair(delta_x_inp, new_index));
    energy.push_back(energy_inp);
    coor.push_back(std::make_shared<std::vector<coor_t> >(coor_inp.data(), coor_inp.data()+coor_inp.size()));
    coor.back()->shrink_to_fit();
    count.push_back(1);
}

template<typename distance_policy>
pele::Array<typename MinimaList<distance_policy>::coor_t> MinimaList<distance_policy>::euclidean_displacement_vector(const index_t idx_i, const index_t idx_j, const index_t nr_points)const
{
    if (nr_points <= 1) throw std::runtime_error("MinimaList<distance_policy>::euclidean_displacement_vector: illegal number of points");
    std::vector<coor_t> res(*coor.at(idx_i));
    res.shrink_to_fit();
    const coor_t del = 1/coor_t(nr_points-1);
    for (index_t i = 0; i < res.size(); ++i){
	res.at(i) -= coor.at(idx_j)->at(i);
	res.at(i) *= del;
    }
    return pele::Array<coor_t>(res).copy();
}

template<typename distance_policy>
pele::Array<typename MinimaList<distance_policy>::coor_t> MinimaList<distance_policy>::euclidean_displacement_vector(const index_t idx_i, pele::Array<coor_t> coords_j, const index_t nr_points)const
{
    if (nr_points <= 1) throw std::runtime_error("MinimaList<distance_policy>::euclidean_displacement_vector: illegal number of points");
    pele::Array<coor_t> res = pele::Array<coor_t>( *coor.at(idx_i) ).copy();
    const coor_t del = 1/coor_t(nr_points-1);
    for (index_t i = 0; i < res.size(); ++i){
	res[i] -= coords_j[i];
	res[i] *= del;
    }
    return res.copy();
}

}//namespace bv
#endif//#ifndef _BV_MINIMA_LIST_H__
