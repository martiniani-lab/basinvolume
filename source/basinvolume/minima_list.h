#ifndef _BV_MINIMA_LIST_H__
#define _BV_MINIMA_LIST_H__

#include <map>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <list>
#include <algorithm>

#include "pele/distance.h"

#include "minimum.h"

namespace bv{

class MinimaList{

public:
    typedef Minimum::energy_t energy_t;
    typedef Minimum::coor_t coor_t;
    typedef Minimum::index_t index_t;
    typedef std::list<Minimum> store_t;
    typedef std::multimap<coor_t,Minimum*> map_t;

private:
    const coor_t tol_delta_x;
    const energy_t tol_energy;
    const coor_t tol_delta_x_element;
    store_t minima_storage;
    map_t minima_order;

public:
    MinimaList(const coor_t tol_delta_x_, const energy_t tol_energy_, const coor_t tol_delta_x_element_):
	tol_delta_x(tol_delta_x_), tol_energy(tol_energy_), tol_delta_x_element(tol_delta_x_element_)
	{}
    /*
     * To make the distance comparison between CheckSameMinimum and here consistent, the tolerances should be set accordingly.
     * */

    index_t nr_distinct_minima()const{return minima_storage.size();}
    std::vector<index_t> nr_minima_visits()const{return property_listing<index_t>(&Minimum::count);}
    std::vector<energy_t> energies()const{return property_listing<energy_t>(&Minimum::energy);}
    std::vector<coor_t> delta_x_listing()const{return property_listing<coor_t>(&Minimum::delta_x);}
    //TODO: consider how the minima should be best accessed to be dumped to the database (via check same minimum class)!

    template<class T>
    std::vector<T> property_listing(T(Minimum::*property)()const)const{
	std::vector<T> result;
	result.reserve(nr_distinct_minima());
	for (store_t::const_iterator i = minima_storage.begin(); i != minima_storage.end(); ++i){
	    result.push_back(((*i).*property)());
	}
	result.shrink_to_fit();
	return result;
    }

    //The following 4 functions are index based and can not be implemented efficently here
    /*
    index_t nr_minimum_visits(const index_t idx)const{return count.at(idx);}
    energy_t get_energy(const index_t idx)const{return energy.at(idx);}
    coor_t get_delta_x(const index_t idx)const{return delta_x.at(idx);}
    pele::Array<coor_t> get_coords(const index_t idx)const{return pele::Array<double>(*coor.at(idx)).copy();}
    */

    bool check_new_minimum(const Minimum& input, pele::Array<coor_t> rattler)
    {
	return check_new_minimum(input.delta_x(), input.energy(), input.coor(), rattler);
    }

    bool check_new_minimum(const coor_t delta_x_inp, const energy_t energy_inp, pele::Array<coor_t> coor_inp, pele::Array<coor_t> rattler)
    {
	// 1. get possible matches for candidate based on delta_x and dtol
	const map_t::const_iterator low = minima_order.lower_bound(delta_x_inp-tol_delta_x);
	const map_t::const_iterator high= minima_order.upper_bound(delta_x_inp+tol_delta_x);
        // 2. check if candidate agrees with any potential match
        for (map_t::const_iterator i = low; i != high; ++i){
	    Minimum*const& this_match = i->second;
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
    /*
     * Once the CheckSameMinimum test (given the used tol) has decided that the found minimum is different
     * this function takes the information on that "neighbouring" minimum and stores it.
     * If the neighbour was already known, we increase its count and return false.
     * If the neighbour was not yet known (visited), we add it to the list of neighbours and return true.
     * This behaviour is analogous to the behaviour of std::map.
     * We are using std::multimap for now to avoid problems due to floating point arithmetic with the keys (delta_x).
     * Presumably, this is not necessary and can be changed with the typedef above.
     * */

    bool agrees_with_input(const Minimum* this_match, const energy_t energy_inp, pele::Array<coor_t> coor_inp, pele::Array<coor_t> rattler)
    {
        //1. check for (scalar) delta_x passed
        //2. check: energy match
        if ( fabs(this_match->energy()-energy_inp) > tol_energy )
    	return false; //failed energy test
        //3. check: coordinate match
        coor_t* it = this_match->coor().data();
        const index_t tmp_N = this_match->coor().size();
        for (index_t i = 0; i < tmp_N; ++i, ++it){
    	if ( rattler[i]*fabs( *it - coor_inp[i] ) > tol_delta_x_element )
    	    return false; //failed coordinate test
        }
        return true; //all tests passed
    }

    void record_duplicate(Minimum*const& this_match)
    {
	this_match->increment_count();
    }

    void record_new_minimum(const map_t::const_iterator insertion_hint, const coor_t delta_x_inp, const energy_t energy_inp, pele::Array<coor_t> coor_inp)
    {
	minima_storage.push_back( Minimum(delta_x_inp, energy_inp, coor_inp) );
	minima_order.insert(insertion_hint, std::make_pair(delta_x_inp, &minima_storage.back()));
    }
};

}//namespace bv
#endif//#ifndef _BV_MINIMA_LIST_H__
