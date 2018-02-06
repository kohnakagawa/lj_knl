#pragma once

class NeighborList final {
  std::vector<int32_t> i_particles_;
  std::vector<int32_t> j_particles_;
  std::vector<int32_t> number_of_partners_;
  std::vector<int32_t> pointer_, pointer2_;
  std::vector<int32_t> sorted_list_;

#ifdef REACTLESS
  const char* cache_f = "pair_all.bin";
#else
  const char* cache_f = "pair_half.bin";
#endif

  void register_pair(const int index1, const int index2) {
    auto i = index1;
    auto j = index2;
    if (i >= j) {
      i = index2;
      j = index1;
    }
    i_particles_.push_back(i);
    j_particles_.push_back(j);
    number_of_partners_[i]++;
#ifdef REACTLESS
    i_particles_.push_back(j);
    j_particles_.push_back(i);
    number_of_partners_[j]++;
#endif
  }

  void sort_pair(const int32_t pn) {
    int pos = 0;
    pointer_[0] = 0;
    for (int i = 0; i < pn; i++) {
      pos += number_of_partners_[i];
      pointer_[i + 1] = pos;
    }
    std::fill_n(pointer2_.begin(), pn + 1, 0);

    const auto s = number_of_pairs();
    assert(pointer_[pn] == s);
    sorted_list_.resize(s, 0);
    for (int32_t k = 0; k < s; k++) {
      const auto i = i_particles_[k];
      const auto j = j_particles_[k];
      const auto index = pointer_[i] + pointer2_[i];
      sorted_list_[index] = j;
      pointer2_[i]++;
    }
  }

  void random_shfl(const int pn) {
    std::mt19937 mt(10);
    for (int i = 0; i < pn; i++) {
      const auto kp = pointer_[i];
      const auto np = number_of_partners_[i];
      std::shuffle(&sorted_list_[kp], &sorted_list_[kp + np], mt);
    }
  }

  void makepair_bruteforce(const int pn,
                           const double sl,
                           const Vec4* q) {
    const auto sl2 = sl * sl;
    for (int i = 0; i < pn - 1; i++) {
      for (int j = i + 1; j < pn; j++) {
        const auto dx = q[i].x - q[j].x;
        const auto dy = q[i].y - q[j].y;
        const auto dz = q[i].z - q[j].z;
        const auto r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < sl2) {
          register_pair(i, j);
        }
      }
    }
  }

public:
  const int32_t* pointer() const {
    return pointer_.data();
  }

  const int32_t* number_of_partners() const {
    return number_of_partners_.data();
  }

  const int32_t* sorted_list() const {
    return sorted_list_.data();
  }

  const int32_t* i_particles() const {
    return i_particles_.data();
  }

  const int32_t* j_particles() const {
    return j_particles_.data();
  }

  int32_t number_of_pairs() const {
    return i_particles_.size();
  }

  bool loadpair(const int pn_est) {
    std::ifstream ifs(cache_f, std::ios::binary);
    if (!ifs) return false;

    int pn = 0, np = 0;
    ifs.read((char*)&pn, sizeof(int));
    ifs.read((char*)&np, sizeof(int));
    if (pn_est != pn) return false;

    std::cout << "Neighbor list is loaded." << std::endl;

    number_of_partners_.resize(pn);
    i_particles_.resize(np);
    j_particles_.resize(np);

    ifs.read((char*)number_of_partners_.data(), sizeof(int)*pn);
    ifs.read((char*)i_particles_.data(), sizeof(int)*np);
    ifs.read((char*)j_particles_.data(), sizeof(int)*np);

    pointer_.resize(pn + 1, 0);
    pointer2_.resize(pn + 1, 0);
    sort_pair(pn);
    random_shfl(pn);

    return true;
  }

  void make_sorted_list(const int pn,
                        const double sl,
                        const Vec4* q) {
    pointer_.resize(pn + 1, 0);
    pointer2_.resize(pn + 1, 0);
    number_of_partners_.resize(pn, 0);

    makepair_bruteforce(pn, sl, q);
    sort_pair(pn);
    random_shfl(pn);
  }

  void savepair(const int pn) const {
    std::ofstream ofs(cache_f, std::ios::binary);
    const auto np = number_of_pairs();
    ofs.write((char*)&pn, sizeof(int));
    ofs.write((char*)&np, sizeof(int));
    ofs.write((char*)number_of_partners(), sizeof(int)*pn);
    ofs.write((char*)i_particles(), sizeof(int)*np);
    ofs.write((char*)j_particles(), sizeof(int)*np);
  }
};
