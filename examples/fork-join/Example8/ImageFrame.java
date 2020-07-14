// =================================================================
//
// File: ImageFrame.java
// Author: Pedro Perez
// Description: This file implements the necessary functions to open 
//				a simple window in Java.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
 
public class ImageFrame {
    
	public static void showImage(String text, Image image) {
        JFrame frame = new JFrame(text);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
 
        JLabel label = new JLabel(new ImageIcon(image));
        frame.getContentPane().add(label, BorderLayout.CENTER);
 
        frame.pack();
        frame.setVisible(true);
    }
}