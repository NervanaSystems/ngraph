FROM centos:7.4.1708

RUN yum -y update
RUN yum -y install zlib-devel
RUN yum -y install ncurses-libs
RUN yum -y install ncurses-devel
RUN yum -y install patch diffutils
RUN yum -y install wget
RUN yum -y install gcc-c++ make
RUN yum -y install git
RUN yum -y install perl-Data-Dumper

# Installing Cmake 3.4
WORKDIR /root
RUN wget https://cmake.org/files/v3.4/cmake-3.4.3.tar.gz --no-check-certificate
RUN tar -xzvf cmake-3.4.3.tar.gz
WORKDIR cmake-3.4.3
RUN ./bootstrap
RUN make -j 8
RUN make install

# Install nGraph in /root/ngraph
COPY . /root/ngraph
RUN mkdir /root/ngraph/build
WORKDIR /root/ngraph/build
RUN cmake ../
RUN make -j 8
RUN make install
