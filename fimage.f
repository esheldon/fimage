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


      function model_f4image_sub4(model,image,nx,ny,
     &xcen,ycen,ixx,ixy,iyy,dosub) result(flag)

      ! Place a model into th 4-byte floating point image using
      ! sub-pixel integration.
      !
      ! We assume the image is initialized to zero in all pixels

      implicit none

      integer*4 model
      integer*4 flag
      integer*4 nx,ny
      real*4 image(nx,ny)
      real*4 ixx,ixy,iyy
      real*4 xcen,ycen

      logical dosub
      real*4 sum

      real*4 expon,detw

      real*4 x,y
      real*4 xl,yl,xx,yy
      real*4 r2
      real*4 w1,w2,w12

      integer*4 i,j, ii, jj

      real val
c      real isum,xsum,ysum,xxsum,yysum,xysum

c      isum=0
c      xsum=0
c      ysum=0
c      xxsum=0
c      yysum=0
c      xysum=0

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

      w1=ixx/detw
      w2=iyy/detw
      w12=ixy/detw

      do i=1,nx
        x=i-xcen
        xl=x-0.375
        do j=1,ny
          y=j-ycen
          yl=y-0.375

          if (dosub) then
            ! integrate over a 4x4 sub pixel grid

            sum=0

            xx=xl
            do ii=1,4
              yy=yl
              do jj=1,4

                r2=xx*xx*w2 + yy*yy*w1 - 2.*xx*yy*w12

                if (model .eq. 1) then
                  expon=0.5*r2
                else if (model .eq. 2) then
                  expon=sqrt(r2)
                else if (model .eq. 3) then
                  expon=7.67*(r2**0.125 - 1)
                endif

                if (expon.le.100.)then
                  val = exp(-expon)
                  sum = sum + val

c                  isum=isum+val
c                  xsum=xsum + val*(xx+xcen)
c                  ysum=ysum + val*(yy+ycen)
c                  xxsum=xxsum + val*xx*xx
c                  xysum=xysum + val*xx*yy
c                  yysum=yysum + val*yy*yy
                endif

                yy = yy + 0.25
              enddo ! loop y sub pixels
              xx = xx + 0.25
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
      


c      print *,"measured <x>:",xsum/isum
c      print *,"measured <y>:",ysum/isum

c      print *,"measured Ixx:",xxsum/isum
c      print *,"measured Ixy:",xysum/isum
c      print *,"measured Iyy:",yysum/isum

      return
      end
      




      function model_f4image_sub8(model,image,nx,ny,
     &xcen,ycen,ixx,ixy,iyy,dosub) result(flag)

      ! Place a model into th 4-byte floating point image using
      ! sub-pixel integration on an 8x8 sub grid
      !
      ! We assume the image is initialized to zero in all pixels

      implicit none

      integer*4 model
      integer*4 flag
      integer*4 nx,ny
      real*4 image(nx,ny)
      real*4 ixx,ixy,iyy
      real*4 xcen,ycen

      logical dosub
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

      w1=ixx/detw
      w2=iyy/detw
      w12=ixy/detw

      do i=1,nx
        x=i-xcen
        xl=x-0.4375
        do j=1,ny
          y=j-ycen
          yl=y-0.4375

          if (dosub) then
            ! integrate over a 4x4 sub pixel grid

            sum=0

            xx=xl
            do ii=1,8
              yy=yl
              do jj=1,8

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

                yy = yy + 0.125
              enddo ! loop y sub pixels
              xx = xx + 0.125
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
      

      function model_f4image_subpixel(model,image,nx,ny,
     &xcen,ycen,ixx,ixy,iyy,nsub) result(flag)

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
      print *,"fimage:  nsub: ",nsub,"stepsize: ",stepsize,
     & " offset: ",offset

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
      

