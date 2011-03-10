c vim: set filetype=fortran et ts=2 sw=2 sts=2 :

      ! place a a model in images with sub-pixel integration. Each
      ! pixel is divided into a nxn grid.  For 4x4
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
      ! the upper left corner is (-3/8, 3/8).
      ! The formulas for working on the sub grid:
      !   stepsize = 1./nsub
      !   offset = (nsub-1)*stepsize/2.
      !

      ! various models are supported.  The first input to each routine
      ! is an integer "model"
      !   1: gaussian
      !   2: exponential
      !   3: devauc profile, e.g. exp(-r^0.25)
      !
      ! In all cases the ixx,ixy,iyy form the covariance matrix.


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
                  expon=sqrt(r2*3.)
                else if (model .eq. 3) then
                  expon=7.67*(r2**0.125 - 1)
                endif

                sum = sum + exp(-expon)

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
              expon=sqrt(r2*3.)
            else if (model .eq. 3) then
              expon=7.67*(r2**0.25 - 1)
            endif

            image(i,j) = exp(-expon)

          endif

        enddo ! y indices
      enddo ! x indices
      
      return
      
      end
      
      function model_f8image(model,image,nx,ny,xcen,ycen,
     &                       ixx,ixy,iyy,nsub) result(flag)

      ! Place a model into th 4-byte floating point image using
      ! sub-pixel integration on an nsubXnsub grid
      !
      ! We assume the image is initialized to zero in all pixels

      implicit none

      integer*4 model
      integer*4 flag
      integer*4 nx,ny
      real*8 image(nx,ny)
      real*8 xcen,ycen
      real*8 ixx,ixy,iyy

      integer*4 nsub
      real*8 stepsize, offset

      real*8 sum

      real*8 expon,detw

      real*8 x,y
      real*8 xl,yl,xx,yy
      real*8 r2
      real*8 w1,w2,w12


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
                  expon=sqrt(r2*3.)
                else if (model .eq. 3) then
                  expon=7.67*(r2**0.125 - 1)
                endif

                sum = sum + exp(-expon)

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
              expon=sqrt(r2*3.)
            else if (model .eq. 3) then
              expon=7.67*(r2**0.25 - 1)
            endif

            image(i,j) = exp(-expon)

          endif

        enddo ! y indices
      enddo ! x indices
      
      return
      
      end
      

