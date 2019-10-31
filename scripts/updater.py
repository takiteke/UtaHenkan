#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable

class FacadeUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.enc, self.dec, self.dis = kwargs.pop('models')
        super(FacadeUpdater, self).__init__(*args, **kwargs)


    def loss_enc(self, enc, x_out, t_out, y_out, lam1=10, lam2=1):
        batchsize,_,w,h = y_out.data.shape
        loss_rec = lam1*(F.mean_absolute_error(x_out, t_out))
        loss_adv = lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        chainer.report({'loss': loss}, enc)
        return loss

    def loss_dec(self, dec, x_out, t_out, y_out, lam1=10, lam2=1):
        batchsize,_,w,h = y_out.data.shape
        loss_rec = lam1*(F.mean_absolute_error(x_out, t_out))
        loss_adv = lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        chainer.report({'loss': loss}, dec)
        return loss
    

    def loss_dis(self, dis, y_in, y_out):
        batchsize,_,w,h = y_in.data.shape
        
        L1 = F.sum(F.softplus(-y_in)) / batchsize / w / h
        L2 = F.sum(F.softplus(y_out)) / batchsize / w / h
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss
    
    def update_core(self):
        enc_optimizer = self.get_optimizer('enc')
        dec_optimizer = self.get_optimizer('dec')
        dis_optimizer = self.get_optimizer('dis')

        enc, dec, dis = self.enc, self.dec, self.dis
        xp = enc.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        in_ch = batch[0][0].shape[0]
        out_ch = batch[0][1].shape[0]
        w_in_h = 512
        w_in_w = 128
        w_out_h = 512
        w_out_w = 128

        x_in = xp.zeros((batchsize, in_ch, w_in_h, w_in_w)).astype("f")
        t_out = xp.zeros((batchsize, out_ch, w_out_h, w_out_w)).astype("f")

        for i in range(batchsize):
            x_in[i,:] = xp.asarray(batch[i][0])
            t_out[i,:] = xp.asarray(batch[i][1])
        x_in = Variable(x_in)

        z = enc(x_in)
        x_out = dec(z)
        
        st_in_h = int(w_in_h / 3)
        st_in_w = int(w_in_w / 4 * 3)
        #x_in = Variable(x_in.data[:,:,:,w_in_st:w_in_w])
        x_in = x_in[:,:,st_in_h*2:w_in_h,st_in_w:w_in_w]
        x_out = x_out[:,:,st_in_h*2:w_in_h,st_in_w:w_in_w]
        t_out = t_out[:,:,st_in_h*2:w_in_h,st_in_w:w_in_w]
        y_fake = dis(x_in, x_out)
        y_real = dis(x_in, t_out)


        enc_optimizer.update(self.loss_enc, enc, x_out, t_out, y_fake)
        for z_ in z:
            z_.unchain_backward()
        dec_optimizer.update(self.loss_dec, dec, x_out, t_out, y_fake)
        x_in.unchain_backward()
        x_out.unchain_backward()
        dis_optimizer.update(self.loss_dis, dis, y_real, y_fake)