#ifndef SILL_SERIALIZE_BOOST_GREGORIAN_DATE_HPP
#define SILL_SERIALIZE_BOOST_GREGORIAN_DATE_HPP

#include <boost/date_time/gregorian/gregorian.hpp>

#include <sill/serialization/serialize.hpp>

namespace sill {
  
  inline oarchive& operator<<(oarchive& a, const boost::gregorian::date& date) {
    if (date.is_special()) {
      a << char(0) << char(date.as_special());
    } else {
      char y1 = date.year() & 0x7f;
      char y2 = date.year() >> 7;
      a << y1 << y2 << char(date.month()) << char(date.day());
    }
    return a;
  }

  inline iarchive& operator>>(iarchive& a, boost::gregorian::date& date) {
    char y1, y2, m, d;
    a >> y1;
    if (y1 == 0) { // special date
      a >> y2;
      date = boost::gregorian::date(boost::gregorian::special_values(y2));
    } else {
      a >> y2 >> m >> d;
      date = boost::gregorian::date(int(y1) | (int(y2) << 7), m, d);
    }
    return a;
  }

}

#endif
