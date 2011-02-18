c vim: set filetype=fortran et ts=2 sw=2 sts=2 :

      ! place a a model in images with sub-pixel integration. Each
      ! pixel is divided into a 4x4 grid.
      !
      !      -----------------
      !      |   |   |   |   |
      !      -----------------
      !      |   |   |   |   |
      !      -------cen-------
      !      |   |   |   |   |
      !      -----------------
      !      |   |   |   |   |
      !      -----------------
      !
      ! If the center of the pixl is as shown, then the center of
      ! the upper left corner is (-3/8, 3/8).  3/8 = 0.375 and
      ! this number is sprinkled throughout the code below
      !

      ! various models are supported.  The first input to each routine
      ! is an integer "model"
      !   1: gaussian
      !   2: exponential
      !   3: devauc profile
      !
      ! In all cases the ixx,ixy,iyy form the covariance matrix, but
      ! note this translates very differently into ellipticity
      ! for each model


      function model_f4image(model,image,nx,ny,xcen,ycen,
     &                       ixx,ixy,iyy,nsub) result(flag)

      ! Place a model into th 4-byte floating point image using
      ! sub-pixel integration on an nsubXnsub grid
      !
      ! We assume the image is initialized to zero in all pixels

      implicit none

      integer*4 model
      integer*4 flag
      integer*4 nx,ny
      real*4 image(nx,ny)
      real*4 xcen,ycen
      real*4 ixx,ixy,iyy

      integer*4 nsub
      real*4 stepsize, offset

      real*4 sum

      real*4 expon,detw

      real*4 x,y
      real*4 xl,yl,xx,yy
      real*4 r2
      real*4 w1,w2,w12


      integer*4 i,j, ii, jj


      flag=0
      expon=0 ! to shut up the warnings

      if (model .ne. 1 .and. model .ne. 2 .and. model .ne. 3) then
        flag=2**0
        return
      endif

      detw=ixx*iyy-ixy*ixy
      if(detw.le.0.)then
        flag=2**1
        return
      endif

      if (nsub <= 0) then
        flag=2**2
        return
      endif

      stepsize = 1./nsub
      offset = (nsub-1)*stepsize/2.
c      print *,"fimage:  nsub: ",nsub,"stepsize: ",stepsize,
c     & " offset: ",offset

      w1=ixx/detw
      w2=iyy/detw
      w12=ixy/detw

      do i=1,nx
        x=i-xcen
        xl=x-offset
        do j=1,ny
          y=j-ycen
          yl=y-offset

          if (nsub > 1) then
            ! integrate over a nsubXnsub pixel grid

            sum=0

            xx=xl
            do ii=1,nsub
              yy=yl
              do jj=1,nsub

                r2=xx*xx*w2 + yy*yy*w1 - 2.*xx*yy*w12

                if (model .eq. 1) then
                  expon=0.5*r2
                else if (model .eq. 2) then
                  expon=sqrt(r2)
                else if (model .eq. 3) then
                  expon=7.67*(r2**0.125 - 1)
                endif

                if (expon.le.100.)then
                  sum = sum + exp(-expon)
                endif

                yy = yy + stepsize
              enddo ! loop y sub pixels
              xx = xx + stepsize
            enddo ! loop x sub pixels

            image(i,j) = sum

          else 
            ! no sub-pixel integration
            r2= x*x*w2 + y*y*w1 - 2.*x*y*w12

            if (model .eq. 1) then
              expon=0.5*r2
            else if (model .eq. 2) then
              expon=sqrt(r2)
            else if (model .eq. 3) then
              expon=7.67*(r2**0.25 - 1)
            endif

            if (expon.le.100.)then
              image(i,j) = exp(-expon)
            endif

          endif

        enddo ! y indices
      enddo ! x indices
      
      return
      
      end
      

